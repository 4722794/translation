#!/bin/bash

# Checkout the specified branch
git checkout temp_sweeps

# Install required Python packages
pip install -r requirements.txt

# Login to wandb
wandb login f604838e6979745cd235989e2e5277b3a8084f95

# Run the main Python script
python main.py
