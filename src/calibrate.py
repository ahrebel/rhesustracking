# calibrate.py

import cv2
import argparse
import numpy as np
from deep_pose_estimation import estimate_pose

def calibrate_video(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_count = 0
    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 30th frame for calibration
        if frame_count % 30 == 0:
            keypoints = estimate_pose(frame)
            all_keypoints.append(keypoints)
            print(f"Frame {frame_count}: Detected keypoints: {keypoints}")

            # Optional: display the frame with keypoints drawn
            for (x, y) in keypoints:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Save the collected keypoints for further calibration work
    np.save(output_file, all_keypoints)
    print(f"Calibration data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate video using DeepPoseKit.")
    parser.add_argument("video", help="Path to the calibration video file.")
    parser.add_argument("--output", default="calibration_data.npy",
                        help="Output file for calibration data.")
    args = parser.parse_args()

    calibrate_video(args.video, args.output)
