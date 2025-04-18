import itertools
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

# Paths
yaml_path = Path("/Users/lucasmckamey/Desktop/Thinking_Folder/Cornell/Senior/Semester_2/RL/P3/cs4789pa3/hyperparameters.yaml")
test_script = "/opt/miniconda3/envs/CS_Reinforcement/bin/python"
test_script_path = "/Users/lucasmckamey/Desktop/Thinking_Folder/Cornell/Senior/Semester_2/RL/P3/cs4789pa3/test_ppo.py"
env_id = "Reacher-v4"
logfile = Path("search_results.log")


# LunarLander-v2:
#   clip_ratio: 0.2
#   ent_coef: 0.01
#   epochs: 300
#   gae_lambda: 0.9
#   gamma: 0.97
#   lr: 0.001
#   minibatch_size: 64
#   num_steps: 256
#   seed: 42
#   update_epochs: 3
#   vf_coef: 0.5


# Hyperparameter search space
param_grid = {
    "minibatch_size": [32, 64, 128],
    "num_steps": [256, 512, 1024],
    "epochs": [50, 100, 300],
    "gae_lambda": [0.9, 0.92, 0.95, 0.97, 0.99],
}

# Load base YAML
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Save updated YAML
def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)

# Run test script
def run_test():
    result = subprocess.run(
        [test_script, test_script_path, f"--env_id={env_id}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout

# Log results
def log_result(config, output):
    with open(logfile, "a") as f:
        f.write(f"--- {datetime.now()} ---\n")
        f.write(f"Config: {config}\n")
        f.write("Output:\n")
        f.write(output)
        f.write("\n\n")

# Cartesian product of all hyperparameter combinations
keys, values = zip(*param_grid.items())
combinations = list(itertools.product(*values))

for combo in combinations:
    config = dict(zip(keys, combo))
    print(f"Running config: {config}")

    # Load YAML
    yaml_data = load_yaml(yaml_path)

    # Update only relevant params for Reacher-v4
    for key, value in config.items():
        yaml_data[env_id][key] = value

    # Save new config
    save_yaml(yaml_data, yaml_path)

    # Run test and log output
    output = run_test()
    log_result(config, output)
