import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Your run path (update this)
ENTITY = "laithzumot"
PROJECT = "huggingface" 
RUN_ID = "yqnbvuec"  # Get this from wandb UI

# Initialize API
api = wandb.Api()

# Get run
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

# 1. Get completions table
table = run.summary["completions"]  # Or "lean_completions"
df = pd.DataFrame(table.data)

print(f"Found {len(df)} completions across {df['step'].nunique()} steps")

# 2. Plot reward distribution
plt.figure(figsize=(14, 6))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(df['step'], df['reward'], alpha=0.6, s=30)
plt.xlabel('Training Step')
plt.ylabel('Reward (0 or 1)')
plt.title('Reward Distribution Over Time')
plt.yticks([0, 1])
plt.grid(alpha=0.3)

# Histogram
plt.subplot(1, 2, 2)
plt.hist(df['reward'], bins=[-0.1, 0.1, 0.9, 1.1], rwidth=0.8, align='mid')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Frequency')
plt.xticks([0, 1])

plt.tight_layout()
plt.savefig('reward_distribution.png')
print("Saved plot to reward_distribution.png")

# 3. Print statistics
print("\nReward Statistics:")
print(f"Total completions: {len(df)}")
print(f"Successful proofs (reward=1): {df['reward'].sum()} ({df['reward'].mean():.1%})")
print(f"Steps with at least one success: {(df.groupby('step')['reward'].sum() > 0).sum()}")