import numpy as np
import torch
import pdb

from ..utils.arrays import to_torch,to_np

VALUE_PLACEHOLDER = 1e6

def make_prefix(discretizer, context, obs, prefix_context=True):
    observation_dim = obs.size
    obs_discrete = discretizer.discretize(obs, subslice=[0, observation_dim])
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)

    if prefix_context:
        prefix = torch.cat(context + [obs_discrete], dim=-1)
    else:
        prefix = obs_discrete

    return prefix

def make_prefix_batch(discretizer, obs_batch, prefix_context=False, context_batch=None):
    """
    obs_batch: shape (B, obs_dim)
    context_batch: 若需要 prefix context，则 shape (B, context_length)
    """
    B, obs_dim = obs_batch.shape

    # 对每条观测做离散化
    obs_discrete = discretizer.discretize(obs_batch, subslice=[0, obs_dim])
    # obs_discrete 形状依然是 (B, obs_dim)；需要转成 long tensor
    obs_discrete = to_torch(obs_discrete, dtype=torch.long)

    # 拼成 prefix
    if prefix_context and (context_batch is not None):
        prefix = torch.cat([context_batch, obs_discrete], dim=1)
    else:
        prefix = obs_discrete

    return prefix  # (B, prefix_length)


def make_prefix_from_snippet(discretizer, snippet_sa, obs_dim, act_dim,prefix_context=False, context_batch=None):
    """
    将一段 (s, a) 序列离散化并拼接成 GPT 的前缀。

    para:
    - discretizer: QuantileDiscretizer
    - snippet_sa: 形如 (B, L, obs_dim + act_dim) 的实数张量/数组
       表示 batch_size=B，每条 snippet 长度=L，每步 (s_t, a_t) 拼在一起
    - prefix_context: 是否再拼上额外的 tokens (比如以前生成的 context)
    - context_batch: 若 prefix_context=True，则 shape (B, some_length)

    return:
    - prefix: shape (B, prefix_length)
      这里 prefix_length = L * (obs_dim + act_dim)  (因为 reward、done 不考虑)
    """
    if isinstance(snippet_sa, torch.Tensor):
        snippet_sa_np = to_np(snippet_sa)
    else:
        snippet_sa_np = snippet_sa  # np.ndarray

    B, L, sa_dim = snippet_sa_np.shape  # sa_dim = obs_dim + act_dim

    # 做离散化
    # snippet_sa_np 形状 (B, L, sa_dim)
    # 需要 reshape => (B*L, sa_dim) 做一次性 discretize
    snippet_flat = snippet_sa_np.reshape(B * L, sa_dim)
    # 调用 discretizer.discretize
    snippet_indices = discretizer.discretize(
        snippet_flat,
        subslice=(0, obs_dim + act_dim)  # 新增 subslice
    )

    #  reshape => (B, L*sa_dim)
    #  discretize(...) 返回 (B*L, ) or (B*L, ?)
    #  但是这里离散化的维度是一条 => snippet_indices shape: (B*L, )
    #  我们要把它变成 (B, L*sa_dim)
    #  所以, discretize() 的每列都离散化 => snippet_indices shape=(B*L, sa_dim')
    #  然后 flatten => (B, L*sa_dim)

    if snippet_indices.ndim == 2:
        # shape (B*L, sa_dim) => reshape => (B, L*sa_dim)
        snippet_indices_2d = snippet_indices.reshape(B, L * sa_dim)
    else:
        # shape (B*L,) => reshape => (B, L)
        snippet_indices_2d = snippet_indices.reshape(B, L)

    # 转成 long tensor
    snippet_indices_2d = to_torch(snippet_indices_2d, dtype=torch.long)

    # 如果 prefix_context=True，则再拼上 context_batch（可选）
    if prefix_context and (context_batch is not None):
        prefix = torch.cat([context_batch, snippet_indices_2d], dim=1)
    else:
        prefix = snippet_indices_2d

    return prefix  # shape (B, prefix_length)

def extract_actions(x, observation_dim, action_dim, t=None):
    assert x.shape[1] == observation_dim + action_dim + 2
    actions = x[:, observation_dim:observation_dim+action_dim]
    if t is not None:
        return actions[t]
    else:
        return actions

def update_context(context, discretizer, observation, action, reward, max_context_transitions):
    """
        context : list of transitions
            [ tensor( transition_dim ), ... ]
    """
    ## use a placeholder for value because input values are masked out by model
    rew_val = np.array([reward, VALUE_PLACEHOLDER])
    transition = np.concatenate([observation, action, rew_val])

    ## discretize transition and convert to torch tensor
    transition_discrete = discretizer.discretize(transition)
    transition_discrete = to_torch(transition_discrete, dtype=torch.long)

    ## add new transition to context
    context.append(transition_discrete)

    ## crop context if necessary
    context = context[-max_context_transitions:]

    return context
