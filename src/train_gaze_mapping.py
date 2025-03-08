#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

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
    # Expects columns: left_corner_x, left_corner_y, right_corner_x, right_corner_y,
    # left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, screen_x, screen_y
    df = pd.read_csv(csv_path)
    X_calib = df[[
        'left_corner_x','left_corner_y','right_corner_x','right_corner_y',
        'left_pupil_x','left_pupil_y','right_pupil_x','right_pupil_y'
    ]].values
    Y_calib = df[['screen_x','screen_y']].values
    return X_calib, Y_calib

def train_eye_to_screen_net(X_calib, Y_calib, epochs=2000, lr=0.001):
    X_tensor = torch.from_numpy(X_calib).float()
    Y_tensor = torch.from_numpy(Y_calib).float()
    
    model = EyeToScreenNet(input_dim=8, hidden_dim=16, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="gaze_mapping_model.pth")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    X_calib, Y_calib = load_calibration_data(args.data)
    model = train_eye_to_screen_net(X_calib, Y_calib, args.epochs, args.lr)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
