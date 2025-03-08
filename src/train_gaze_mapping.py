#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# Define a shallow neural network for 8 input features -> 2 outputs (screen_x, screen_y)
class EyeToScreenNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super(EyeToScreenNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

def load_calibration_data(csv_path):
    """
    Loads calibration data from a CSV file. The file must contain columns:
      left_corner_x, left_corner_y, right_corner_x, right_corner_y,
      left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
      screen_x, screen_y
    Returns:
        X_calib: numpy array of shape (N, 8)
        Y_calib: numpy array of shape (N, 2)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Calibration data file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)
    
    required_columns = [
        'left_corner_x', 'left_corner_y',
        'right_corner_x', 'right_corner_y',
        'left_pupil_x', 'left_pupil_y',
        'right_pupil_x', 'right_pupil_y',
        'screen_x', 'screen_y'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in '{csv_path}'.")

    # Build the input and output arrays
    X_calib = df[[
        'left_corner_x', 'left_corner_y',
        'right_corner_x', 'right_corner_y',
        'left_pupil_x',  'left_pupil_y',
        'right_pupil_x', 'right_pupil_y'
    ]].values
    
    Y_calib = df[['screen_x', 'screen_y']].values
    return X_calib, Y_calib

def train_eye_to_screen_net(X_calib, Y_calib, num_epochs=2000, learning_rate=0.001):
    """
    Trains the neural network using the provided calibration data.
    Returns the trained model.
    """
    X_tensor = torch.from_numpy(X_calib).float()
    Y_tensor = torch.from_numpy(Y_calib).float()
    
    model = EyeToScreenNet(input_dim=8, hidden_dim=16, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training the EyeToScreenNet...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

def predict_screen_coordinates(model, landmark_vector):
    """
    Predict screen coordinates from an 8-element landmark vector.
    
    Args:
        model: The trained EyeToScreenNet.
        landmark_vector (array-like): [corner_left_x, corner_left_y, corner_right_x, corner_right_y,
                                       left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y]
    Returns:
        numpy.ndarray: Predicted screen coordinates [x, y]
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(landmark_vector, dtype=torch.float32).unsqueeze(0)
        prediction = model(input_tensor)
    return prediction.squeeze().numpy()

def main():
    parser = argparse.ArgumentParser(description="Train the Eye-to-Screen mapping model")
    parser.add_argument("--data", required=True, help="Path to the calibration CSV file")
    parser.add_argument("--output", default="eye_to_screen_model.pth", help="Output path for the trained model")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    # Load calibration data
    X_calib, Y_calib = load_calibration_data(args.data)
    print(f"Loaded calibration data with shapes: X={X_calib.shape}, Y={Y_calib.shape}")
    
    # Train the network
    model = train_eye_to_screen_net(X_calib, Y_calib, num_epochs=args.epochs, learning_rate=args.lr)
    
    # Save the trained model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model trained and saved to '{args.output}'")

if __name__ == "__main__":
    main()
