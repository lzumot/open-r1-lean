import wandb
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURE THESE ---
ENTITY = "laithzumot"
PROJECT = "huggingface" 
RUN_ID = "yqnbvuec"  # ✅ This is from your URL
# -----------------------

# Initialize API
api = wandb.Api()

# Get run
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

# Extract completions table
try:
    table = run.summary["completions"]
except KeyError:
    table = run.summary["lean_completions"]
    
df = pd.DataFrame(table.data)

print(f"✅ Found {len(df)} completions across {df['step'].nunique()} steps")

# Plot
plt.figure(figsize=(14, 6))

# Reward scatter plot
plt.subplot(1, 2, 1)
plt.scatter(df['step'], df['reward'], alpha=0.6, s=30, color='steelblue')
plt.xlabel('Training Step')
plt.ylabel('Reward (0 or 1)')
plt.title('Lean Proof Rewards Over Time')
plt.yticks([0, 1])
plt.grid(alpha=0.3)

# Histogram
plt.subplot(1, 2, 2)
plt.hist(df['reward'], bins=[-0.1, 0.1, 0.9, 1.1], rwidth=0.8, color='darkorange', align='mid')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Frequency')
plt.xticks([0, 1])

plt.tight_layout()
plt.savefig('/home/lyz/repos/open-r1-lean/past_runs/run_3_algebra/reward_distribution.png', dpi=150)
print("\n📊 Saved plot to: reward_distribution.png")

# Statistics
print("\n📈 Statistics:")
print(f"Total completions: {len(df):,}")
print(f"Successful proofs: {df['reward'].sum():,} ({df['reward'].mean():.1%})")
print(f"Steps with ≥1 success: {(df.groupby('step')['reward'].sum() > 0).sum()}")