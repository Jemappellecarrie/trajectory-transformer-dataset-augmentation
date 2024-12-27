import os
import json
import numpy as np
import torch

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import sample_n
from trajectory.search import make_prefix_batch



@torch.no_grad()
def generate_batch_trajectories(args, gpt, dataset, env, preprocess_fn, batch_size=16):
    """
    一次并行生成 batch_size 条 trajectory
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
                # 到最后一个 step, next_obs 可设置成自身或者另行处理
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
# 3) 主入口：批量生成 1000 条
########################################
def main():
    parser = utils.Parser()
    parser.dataset = 'walker2d-medium-expert-v2'
    parser.config = 'config.offline'
    args = parser.parse_args('plan')

    # 加载 dataset, model
    dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath, 'data_config.pkl')
    gpt, gpt_epoch = utils.load_model(
        args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device
    )
    env = datasets.load_environment(args.dataset)
    preprocess_fn = datasets.get_preprocess_fn(env.name)

    # 一次生成 batch_size 条，多次累积到 1000
    total_trajectories_needed = 1000
    batch_size = 64
    all_trajectories = []

    while len(all_trajectories) < total_trajectories_needed:
        batch_trajs = generate_batch_trajectories(args, gpt, dataset, env, preprocess_fn, batch_size=batch_size)
        all_trajectories.extend(batch_trajs)

    # 如果多了，就切掉
    all_trajectories = all_trajectories[:total_trajectories_needed]

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
        os.path.join(save_dir, "gpt_trajectories.npz"),
        observations=all_obs,
        next_observations=all_next_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
        timeouts=all_timeouts
    )
    print(f"[INFO] GPT trajectories saved at {os.path.join(save_dir, 'gpt_trajectories.npz')}")


if __name__ == "__main__":
    main()