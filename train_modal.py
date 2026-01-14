from modal import App, Image, Volume, Secret, gpu
import datetime
import modal

EXPERIMENT_NAME = f"Lean-GRPO-V2-7B-Algebra-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

# Configuration - easy to edit
GPU_COUNT = 1  # Change to 8 for full run
WORK_DIR = "/app"
CHECKPOINT_DIR = f"{WORK_DIR}/checkpoints"

# Modal setup
app = App("openr1-lean-training")
volume = Volume.from_name("openr1-checkpoints", create_if_missing=True)

# ✅ Use NVIDIA PyTorch image (has python symlink + CUDA pre-configured)
image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .apt_install("git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "pip install modal",
    )
    .dockerfile_commands([f"COPY . {WORK_DIR}"])
    .workdir(WORK_DIR)
    .run_commands(
        "uv venv openr1 --python 3.11",
        ". openr1/bin/activate && uv pip install --upgrade pip",
        "git config --global url.https://github.com/.insteadOf git@github.com:",  # HTTPS fix
        ". openr1/bin/activate && uv pip install vllm==0.7.2",
        ". openr1/bin/activate && uv pip install setuptools",
        ". openr1/bin/activate && uv pip install flash-attn --no-build-isolation",
        "GIT_LFS_SKIP_SMUDGE=1 . openr1/bin/activate && uv pip install -e '.[dev]'"
    )
)


@app.function(
    image=image,
    gpu=f"H100:{GPU_COUNT}",
    volumes={CHECKPOINT_DIR: volume},
    secrets=[Secret.from_name("wandb-secret")],
    timeout=86400,  # 24 hours
    cpu=8,
    memory=32768,  # 32GB RAM
    #spot=True,  # Enable spot instances
)
def train():
    import os
    import subprocess
    
    # Mount checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Activate venv
    activate_cmd = ". /app/openr1/bin/activate && "
    
    # Run training  
    cmd = (
    f"{activate_cmd} ACCELERATE_LOG_LEVEL=info accelerate launch "
    f"--config_file recipes/accelerate_configs/zero2.yaml "
    f"--num_processes={GPU_COUNT} src/open_r1/grpo.py "
    f"--config past_runs/run_3_algebra/myconfig.yaml "
    f"--output_dir {CHECKPOINT_DIR}/{EXPERIMENT_NAME}"
    )
    subprocess.run(cmd, shell=True, check=True)

@app.local_entrypoint()
def main():
    train.remote()