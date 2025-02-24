#!/bin/bash

echo "Creating virtual environment..."
conda install -c conda-forge pytables
python3 -m venv venv
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate the environment with 'source venv/bin/activate'"
