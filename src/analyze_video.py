# analyze_video.py

import cv2
import argparse
from deep_pose_estimation import estimate_pose

def analyze_video(video_path, output_video=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = estimate_pose(frame)

        # Draw keypoints on the frame
        for (x, y) in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.imshow("Pose Estimation", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze video using DeepPoseKit pose estimation."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--output", help="Path to save the output video with pose annotations.")
    args = parser.parse_args()

    analyze_video(args.video, args.output)
