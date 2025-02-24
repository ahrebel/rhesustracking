#!/bin/bash

echo "Creating (or updating) a virtual environment for pip dependencies..."
python3 -m venv venv
source venv/bin/activate
echo "Installing pip dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate the environment with 'source venv/bin/activate' within your Conda environment."
