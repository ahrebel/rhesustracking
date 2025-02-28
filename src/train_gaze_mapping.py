# train_gaze_mapping.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import argparse

def train_gaze_mapping(data_path, output_model='data/trained_model/gaze_mapping_model.pkl'):
    """
    Trains a regression model to map raw eye features to screen coordinates.
    Expected CSV format: columns for 'raw_eye_x', 'raw_eye_y', 'head_roll', 'screen_x', 'screen_y'
    """
    df = pd.read_csv(data_path)
    # Features: raw eye coordinate and head roll.
    X = df[['raw_eye_x', 'raw_eye_y', 'head_roll']].values
    # Targets: screen coordinates.
    y = df[['screen_x', 'screen_y']].values
    model = LinearRegression()
    model.fit(X, y)
    with open(output_model, 'wb') as f:
        pickle.dump(model, f)
    print("Gaze mapping model saved to", output_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--output', type=str, default='data/trained_model/gaze_mapping_model.pkl', help='Output model path')
    args = parser.parse_args()
    train_gaze_mapping(args.data, args.output)
