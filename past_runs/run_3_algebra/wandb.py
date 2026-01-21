import wandb

api = wandb.Api()
run = api.run("laithzumot/huggingface/Lean-GRPO-V2-7B-Al-1")  # Your run path

# Get the completions table as a pandas DataFrame
table = run.summary.get("completions")
df = table.data

# Or download the full history
history = run.history(keys=["rewards/lean", "step"])  # Scalar mean rewards

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(df['step'], df['reward'], alpha=0.6, s=20)
plt.xlabel('Training Step')
plt.ylabel('Reward (0 or 1)')
plt.title('Lean Proof Rewards Over Time (32 problems)')
plt.yticks([0, 1])
plt.grid(alpha=0.3)
plt.show()

plt.hist(df['reward'], bins=[-0.1, 0.1, 0.9, 1.1], rwidth=0.8)
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Distribution (32 problems)')

df['reward_mean_10'] = df.groupby('step')['reward'].transform('mean').rolling(10).mean()
plt.plot(df['step'].unique(), df['reward_mean_10'].unique())
plt.xlabel('Step')
plt.ylabel('Mean Reward')

