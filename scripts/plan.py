import os
import json
import numpy as np
import torch

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import sample_n
from trajectory.search import make_prefix_batch, make_prefix_from_snippet


@torch.no_grad()
def generate_batch_trajectories_with_offline_prefix(
        args, gpt, dataset, offline_data, snippet_len=5, batch_size=16
):
    """
    一次并行生成 batch_size 条 trajectory，prefix 来自 offline_data 的 snippet
    并对 GPT 生成的第1步 next state 与 offline_data 的真实 next state 做欧式距离测量

    offline_data:
       dict or npz with keys: 'observations', 'actions', ...
       shape: (N, obs_dim), (N, act_dim) ...
    snippet_len: 从 offline_data 里截取多长 (s,a) 序列当 prefix，这里是5步（s，a）

    return : trajectories (list of dict), len = batch_size
             distances (list of float), 对应每条 snippet 的第1步距离
    """
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = obs_dim + act_dim + 2  # reward + done

    # 准备 snippet
    # 随机采样 B 条 snippet，每条 snippet 长 snippet_len
    N = offline_data['observations'].shape[0]  # total transitions

    # 要保证 snippet_len <= N
    snippet_sa_list = []

    #用于debug state的分布
    snippet_obs_list = []

    start_indices = [] # 记录每条 snippet 的起点 i, 用于后续做对比
    for _ in range(batch_size):
        # 随机起点 i
        i = np.random.randint(0, N - snippet_len)
        start_indices.append(i)

        # 组装 snippet
        sa_chunk = []
        for t in range(snippet_len):
            s_t = offline_data['observations'][i + t]
            a_t = offline_data['actions'][i + t]

            snippet_obs_list.append(s_t)
            # 拼 (s_t, a_t)
            sa_chunk.append(np.concatenate([s_t, a_t], axis=0))

        sa_chunk = np.stack(sa_chunk, axis=0)  # shape (snippet_len, obs_dim+act_dim)
        snippet_sa_list.append(sa_chunk)

        snippet_obs_array = np.array(snippet_obs_list)

    snippet_sa_batch = np.stack(snippet_sa_list, axis=0)  # shape (B, snippet_len, obs_dim+act_dim)

    # 构建 prefix
    discretizer = dataset.discretizer
    prefix = make_prefix_from_snippet(
        discretizer,
        snippet_sa_batch,
        obs_dim=dataset.observation_dim,
        act_dim=dataset.action_dim
    )
    prefix = prefix.to(args.device)

    # 做一次性采样
    T = args.horizon
    N_tokens = T * transition_dim  # 需要生成的token数量
    sequence, probs = sample_n(gpt, prefix, N_tokens)

    prefix_len = prefix.shape[1]
    generated_tokens = sequence[:, prefix_len:]  # (B, N_tokens)
    # reshape => (B, T, transition_dim)
    generated_tokens = generated_tokens.reshape(batch_size, T, transition_dim)

    # reconstruct & parse => trajectory
    trajectories = []
    distances = []  # 用于存储 (GPT第1步) 和 (offline真实第1步) 的欧式距离(绝对误差和相对误差)
    rel_errors = []
    for b in range(batch_size):
        seq_b = generated_tokens[b]  # shape (T, transition_dim)
        seq_recon = discretizer.reconstruct(seq_b)  # => (T, transition_dim) in continuous

        # GPT 生成的 第1步 next state => seq_recon[0, :obs_dim]
        gpt_next_s0 = seq_recon[0, :obs_dim]

        # 对比 offline 的真实 next state => offline_data['observations'][ i_start + snippet_len ]
        i_start = start_indices[b]

        # 先看有没有越界
        if (i_start + snippet_len) < N:

            #绝对误差
            real_next_s0 = offline_data['observations'][i_start + snippet_len]
            dist = np.linalg.norm(gpt_next_s0 - real_next_s0)

            #相对误差
            denom = np.linalg.norm(real_next_s0) + 1e-8
            rel_error = dist / denom
        else:
            dist = np.nan  # 或者不计入

        distances.append(dist)
        rel_errors.append(rel_error)

        # 拆分整条轨迹
        obs_list, act_list, rew_list, next_obs_list, done_list, timeouts_list = [], [], [], [], [], []
        for i in range(T):
            transition = seq_recon[i]
            idx = 0
            o_ = transition[idx: idx + obs_dim];
            idx += obs_dim
            a_ = transition[idx: idx + act_dim];
            idx += act_dim
            r_ = transition[idx];
            idx += 1
            d_ = transition[idx];
            idx += 1

            if i < T - 1:
                next_o_ = seq_recon[i + 1][:obs_dim]
            else:
                next_o_ = o_

            obs_list.append(o_.tolist())
            act_list.append(a_.tolist())
            rew_list.append(float(r_))
            next_obs_list.append(next_o_.tolist())
            done_list.append(bool(d_))
            timeouts_list.append(False)

        traj_dict = {
            "observations": obs_list,
            "actions": act_list,
            "rewards": rew_list,
            "next_observations": next_obs_list,
            "terminals": done_list,
            "timeouts": timeouts_list
        }
        trajectories.append(traj_dict)

    return trajectories, distances, rel_errors, snippet_obs_array

@torch.no_grad()
def generate_batch_trajectories(args, gpt, dataset, env, preprocess_fn, batch_size=16):
    """
    一次并行生成 batch_size 条 trajectory
    不包括使用上下文生成轨迹，用作对比实验
    返回: trajectories (list of dict), len = batch_size
    dict 里包含 'observations', 'actions', 'rewards', 'next_observations', 'terminals', 'timeouts'
    """

    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = obs_dim + act_dim + 2  # (reward+done)

    # 1) 收集 batch_size 个初始观测
    init_obs_list = []
    for _ in range(batch_size):
        obs = env.reset()
        obs = preprocess_fn(obs)
        init_obs_list.append(obs)
    init_obs_batch = np.stack(init_obs_list, axis=0)  # shape (B, obs_dim)

    # 2) 获取 prefix
    discretizer = dataset.discretizer
    prefix = make_prefix_batch(discretizer, init_obs_batch)  # (B, prefix_length) 这里 prefix_length = obs_dim
    prefix = prefix.to(args.device)

    # 3) 一次性采样
    N = transition_dim * args.horizon
    sequence, probs = sample_n(gpt, prefix, N)  # => shape (B, prefix_length + N)
    prefix_len = prefix.shape[1]
    generated_tokens = sequence[:, prefix_len:]  # => (B, N)

    # 4) reshape => (B, T, tokens_per_transition)
    T = args.horizon  # horizon steps
    generated_tokens = generated_tokens.reshape(batch_size, T, transition_dim)

    # 5) reconstruct & 解析
    trajectories = []
    for b in range(batch_size):
        seq_b = generated_tokens[b]  # (T, transition_dim)
        seq_recon = discretizer.reconstruct(seq_b)  # => (T, transition_dim) in continuous space

        # 拆分
        obs_list = []
        act_list = []
        rew_list = []
        next_obs_list = []
        done_list = []
        timeouts_list = []

        for i in range(T):
            transition = seq_recon[i]  # (obs_dim + act_dim + 2)

            idx = 0
            o_ = transition[idx: idx + obs_dim]
            idx += obs_dim
            a_ = transition[idx: idx + act_dim]
            idx += act_dim
            r_ = transition[idx]
            idx += 1
            d_ = transition[idx]
            idx += 1

            if i < T - 1:
                next_obs_ = seq_recon[i + 1][:obs_dim]
            else:
                # 到最后一个 step, next_obs 可设置成自身
                next_obs_ = o_

            obs_list.append(o_.tolist())
            act_list.append(a_.tolist())
            rew_list.append(float(r_))
            next_obs_list.append(next_obs_.tolist())
            done_list.append(bool(d_))
            timeouts_list.append(False)

        traj_dict = {
            "observations": obs_list,
            "actions": act_list,
            "rewards": rew_list,
            "next_observations": next_obs_list,
            "terminals": done_list,
            "timeouts": timeouts_list
        }
        trajectories.append(traj_dict)

    return trajectories


########################################
# 3) 主入口：批量生成 1000 条（每条5步）
########################################

def main():
    parser = utils.Parser()
    parser.dataset = 'walker2d-medium-expert-v2'
    parser.config = 'config.offline'
    args = parser.parse_args('plan')

    dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath, 'data_config.pkl')

    gpt, gpt_epoch = utils.load_model(
        args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device
    )

    env = datasets.load_environment(args.dataset)
    d4rl_data = env.get_dataset()

    offline_data = {
        'observations': d4rl_data['observations'],  # shape (N, obs_dim)
        'actions': d4rl_data['actions'],  # shape (N, act_dim)
    }

    #检查这些state的norm分布
    # 取出观测数组
    obs_array = offline_data['observations']  # 形状 (N, obs_dim)

    # 对每条观测做 norm
    # norms[i] = sqrt( sum_{dim}( obs_array[i, dim]^2 ) )
    norms = np.linalg.norm(obs_array, axis=1)  # shape (N, )

    # 计算 mean, max, min, percentile
    mean_norm = norms.mean()
    max_norm = norms.max()
    min_norm = norms.min()
    p90_norm = np.percentile(norms, 90)  # 90分位数

    print("[DEBUG] offline_data 'observations' norm stats:")
    print(f"    min: {min_norm:.4f}   max: {max_norm:.4f}")
    print(f"    mean: {mean_norm:.4f}   p90: {p90_norm:.4f}")


    # 一次生成 batch_size 条，多次累积到 1000
    total_trajectories_needed = 1000
    batch_size = 64
    all_trajectories = []
    all_distances = []
    all_rel_errors = []

    global_snippet_obs = []

    while len(all_trajectories) < total_trajectories_needed:

        # 这里有一个复杂的关系，snippest len的长度和gpt模型的block size以及horizon参数的大小有关系，之前生成轨迹的时候，horizon大小为9
        # 但假如这里继续使用horizon为9，那么留给prefix的tokens数量将不够，最多只能使用一步的（s，a）作为前缀上下文，
        # 所以，为了更好的达成上下文丰富的条件，这里把horizon改为5，经过计算，最多支持5步的（s，a）作为前缀。
        # 计算的过程为：在我们所在的这个测试环境下，obs dim是16，act dim是7， 再加上reward和done，总共是25
        # 也就是说，一步就需要25，轨迹总共9步，9*25是225， gpt的block size是249，也就是说我们还剩249-225 = 24的剩余空间
        # 去make prefix时看上下文，而看上下文需要 16+7（state，action）= 23 的空间，所以最多只能看一步的（s，a），snippet_len 最多等于 1
        # 为了丰富上下文，这里经过计算，把horizon设为5，snippet_len设为5较为合理。

        batch_trajs, batch_dists, rel_errors, snippet_obs_array = generate_batch_trajectories_with_offline_prefix(args, gpt, dataset, offline_data,
                                                                      snippet_len=5, batch_size=batch_size)

        all_trajectories.extend(batch_trajs)
        all_distances.extend(batch_dists)
        all_rel_errors.extend(rel_errors)
        global_snippet_obs.append(snippet_obs_array)

    # 如果多了，就切掉（这里可能是欧式距离测量误差过大的因素之一，截断后的分布不一定合理）
    all_trajectories = all_trajectories[:total_trajectories_needed]
    all_distances = all_distances[:total_trajectories_needed]
    all_rel_errors = all_rel_errors[:total_trajectories_needed]

    avg_dist = np.nanmean(all_distances)
    print(f"[INFO] Average 1-step next-state distance = {avg_dist:.4f}")

    avg_rel = np.nanmean(all_rel_errors)
    print(f"[INFO] Average 1-step next-state *relative* error = {avg_rel:.4f}")


    global_snippet_obs = np.concatenate(global_snippet_obs, axis=0)
    snippet_norms = np.linalg.norm(global_snippet_obs, axis=1)
    print(
        f"[INFO] snippet obs norm => min={snippet_norms.min():.4f}, max={snippet_norms.max():.4f}, "
        f"mean={snippet_norms.mean():.4f}, p90={np.percentile(snippet_norms, 90):.4f}")

    # flatten => 存成 npz
    all_obs = []
    all_next_obs = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_timeouts = []

    for traj in all_trajectories:
        all_obs.extend(traj["observations"])
        all_next_obs.extend(traj["next_observations"])
        all_actions.extend(traj["actions"])
        all_rewards.extend(traj["rewards"])
        all_terminals.extend(traj["terminals"])
        all_timeouts.extend(traj["timeouts"])

    all_obs = np.array(all_obs, dtype=np.float32)
    all_next_obs = np.array(all_next_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32).reshape(-1, 1)
    all_terminals = np.array(all_terminals, dtype=np.float32).reshape(-1, 1)
    all_timeouts = np.array(all_timeouts, dtype=np.float32).reshape(-1, 1)

    save_dir = args.savepath
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "gpt_trajectories_with_prefix_test.npz"),
        observations=all_obs,
        next_observations=all_next_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
        timeouts=all_timeouts
    )
    print(f"[INFO] GPT trajectories saved at {os.path.join(save_dir, 'gpt_trajectories_with_prefix_test.npz')}")


if __name__ == "__main__":
    main()
