import cv2

def get_video_frames(video_path):
    """
    Generator that yields frames from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
