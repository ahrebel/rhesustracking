# kalman_filter.py
import numpy as np
import cv2

class KalmanFilter2D:
    def __init__(self):
        # 4 state variables: x, y, dx, dy; 2 measurement variables: x, y
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, coord):
        measurement = np.array([[np.float32(coord[0])],
                                [np.float32(coord[1])]])
        if not self.initialized:
            # Initialize state vector with the first measurement
            self.kalman.statePre = np.array([[measurement[0][0]],
                                             [measurement[1][0]],
                                             [0],
                                             [0]], np.float32)
            self.initialized = True

        # Correct with measurement and predict the next state
        self.kalman.correct(measurement)
        predicted = self.kalman.predict()
        smoothed_x, smoothed_y = predicted[0][0], predicted[1][0]
        return (smoothed_x, smoothed_y)
