# train_dlc_model.py
#!/usr/bin/env python
"""
This script trains the DeepLabCut model using the specified configuration.
Make sure you have prepared your labeled data and the DLC project configuration.
"""

import deeplabcut
import os
import yaml

def load_model_config(config_path="config/model_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def train_model():
    # Path to the DeepLabCut project folder (assumed to be in data/trained_model/)
    project_path = os.path.join("data", "trained_model")
    if not os.path.exists(project_path):
        os.makedirs(project_path, exist_ok=True)
    # The DLC config file is assumed to be at project_path/config.yaml.
    config_path = os.path.join(project_path, "config.yaml")
    if not os.path.exists(config_path):
        print(f"DeepLabCut config file not found at {config_path}. Please create a DLC project first.")
        return
    
    # Start training the network (adjust parameters as needed)
    print("Starting training of DeepLabCut model...")
    deeplabcut.train_network(config_path, shuffle=1, displayiters=100, saveiters=500)
    print("Training complete.")

if __name__ == "__main__":
    train_model()
