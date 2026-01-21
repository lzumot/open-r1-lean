import wandb
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# --- CONFIGURE THESE ---
ENTITY = "laithzumot"
PROJECT = "huggingface" 
RUN_ID = "yqnbvuec"
# -----------------------

# Initialize API
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/runs/{RUN_ID}")

# Get the table object (this is just a reference)
table_ref = run.summary["completions"]

# DOWNLOAD THE ACTUAL TABLE FILE FROM W&B ARTIFACTS
# The path is in _json_dict['path']
table_path = table_ref._json_dict['path']

# Download file to temp location
run.file(table_path).download(replace=True)

# Load the JSON
with open(table_path, 'r') as f:
    table_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(table_data['data'], columns=table_data['columns'])

print(f"✅ Successfully loaded {len(df)} completions")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# Plotting
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['step'], df['reward'], alpha=0.6, s=30, color='steelblue')
plt.xlabel('Training Step')
plt.ylabel('Reward (0 or 1)')
plt.title('Lean Proof Rewards Over Time (32 problems)')
plt.yticks([0, 1])
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(df['reward'], bins=[-0.1, 0.1, 0.9, 1.1], rwidth=0.8, color='darkorange', align='mid')
plt.xlabel('Reward')
plt.ylabel('Count')
plt.title('Reward Frequency')
plt.xticks([0, 1])

plt.tight_layout()
output_path = '/home/lyz/repos/open-r1-lean/past_runs/run_3_algebra/reward_distribution.png'
plt.savefig(output_path, dpi=150)
print(f"\n📊 Saved plot to: {output_path}")

# Statistics
print("\n📈 Statistics:")
print(f"Total completions: {len(df):,}")
print(f"Successful proofs: {df['reward'].sum():,} ({df['reward'].mean():.1%})")
print(f"Steps with ≥1 success: {(df.groupby('step')['reward'].sum() > 0).sum()}")

# Cleanup
os.remove(table_path)