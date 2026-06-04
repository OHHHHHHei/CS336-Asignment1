import json
import os

def save_config(config, path):
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def log_metrics(metrics, path):
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, "metrics.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")