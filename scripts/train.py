import os
import numpy as np
import torch
import pdb

import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.models.transformers import GPT

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-expert-v2'
    config: str = 'config.offline'
    merged_file: str = ''

args = Parser().parse_args('train')

# 如果 merged_file 不为空，则加载 .npz
external_dict = None
if args.merged_file and os.path.isfile(args.merged_file):
    print(f"[INFO] Loading merged dataset from: {args.merged_file}")
    data_npz = np.load(args.merged_file)
    external_dict = {
        "observations": data_npz["observations"],       # shape (X, obs_dim)
        "actions": data_npz["actions"],                 # shape (X, act_dim)
        "next_observations": data_npz["next_observations"],
        "rewards": data_npz["rewards"],
        "terminals": data_npz["terminals"],
        "realterminals": data_npz["realterminals"],
    }
    print(f"[INFO] external_dict loaded: {len(external_dict['observations'])} transitions.")
else:
    print("[INFO] No merged_file provided (or file not found). Will use D4RL dataset from env.")
    external_dict = None

# 加载环境 (如果 external_dict=None, 则走原生 D4RL 的加载逻辑)
env = datasets.load_environment(args.dataset)

# 计算 sequence_length
sequence_length = args.subsampled_sequence_length * args.step

# 构建 DiscretizedDataset
dataset_config = utils.Config(
    datasets.DiscretizedDataset,
    savepath=(args.savepath, 'data_config.pkl'),
    env=args.dataset,
    external_dict=external_dict,
    N=args.N,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    discretizer=args.discretizer,
)

dataset = dataset_config()

obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim

block_size = sequence_length * transition_dim - 1
print(f"[INFO] Dataset size: {len(dataset)} | Joined dim: {transition_dim} | block_size: {block_size}")

# 构建 GPT 配置
model_config = utils.Config(
    GPT,
    savepath=(args.savepath, 'model_config.pkl'),
    vocab_size=args.N,
    block_size=block_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd * args.n_head,
    observation_dim=obs_dim,
    action_dim=act_dim,
    transition_dim=transition_dim,
    action_weight=args.action_weight,
    reward_weight=args.reward_weight,
    value_weight=args.value_weight,
    embd_pdrop=args.embd_pdrop,
    resid_pdrop=args.resid_pdrop,
    attn_pdrop=args.attn_pdrop,
)

model = model_config()
model.to(args.device)

# Trainer 配置
warmup_tokens = len(dataset) * block_size
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1,
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    num_workers=0,
    device=args.device,
)
trainer = trainer_config()

# 训练循环
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
save_freq = max(1, n_epochs // args.n_saves)

for epoch in range(n_epochs):
    print(f"\nEpoch: {epoch}/{n_epochs} | {args.dataset} | {args.exp_name}")

    trainer.train(model, dataset)

    save_epoch = (epoch + 1) // save_freq * save_freq
    statepath = os.path.join(args.savepath, f"state_{save_epoch}.pt")
    print(f"[INFO] Saving model to {statepath}")

    state = model.state_dict()
    torch.save(state, statepath)
