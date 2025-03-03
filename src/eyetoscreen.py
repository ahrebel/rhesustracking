import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Define the shallow neural network
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

def load_calibration_data(csv_path='calibration_data.csv'):
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
        raise FileNotFoundError(f"Calibration data file '{csv_path}' not found. Please provide calibration data.")
    
    df = pd.read_csv(csv_path)
    required_columns = ['left_corner_x', 'left_corner_y', 'right_corner_x', 'right_corner_y',
                        'left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y',
                        'screen_x', 'screen_y']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in calibration data.")
    
    # Build the input and output arrays
    X_calib = df[['left_corner_x', 'left_corner_y', 
                  'right_corner_x', 'right_corner_y', 
                  'left_pupil_x', 'left_pupil_y', 
                  'right_pupil_x', 'right_pupil_y']].values
    Y_calib = df[['screen_x', 'screen_y']].values
    return X_calib, Y_calib

def train_eye_to_screen_net(X_calib, Y_calib, num_epochs=2000, learning_rate=0.001):
    """
    Trains the neural network using the provided calibration data.
    Returns the trained model.
    """
    X_tensor = torch.from_numpy(X_calib).float()
    Y_tensor = torch.from_numpy(Y_calib).float()
    
    model = EyeToScreenNet()
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
        landmark_vector (array-like): [left_corner_x, left_corner_y, right_corner_x, right_corner_y,
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
    # Load calibration data from CSV
    try:
        X_calib, Y_calib = load_calibration_data('calibration_data.csv')
        print("Loaded calibration data with shapes:", X_calib.shape, Y_calib.shape)
    except Exception as e:
        print("Error loading calibration data:", e)
        return
    
    # Train the network using the calibration data
    model = train_eye_to_screen_net(X_calib, Y_calib)
    
    # Save the trained model to disk for later use
    model_path = 'eye_to_screen_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved to '{model_path}'")
    
    # Example: predict screen coordinates using a calibration sample (or new sample if available)
    sample = X_calib[0]  # Using the first sample for demonstration
    predicted_screen = predict_screen_coordinates(model, sample)
    print("Predicted screen coordinates for first sample:", predicted_screen)

if __name__ == "__main__":
    main()
