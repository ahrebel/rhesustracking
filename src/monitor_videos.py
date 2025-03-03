import time
import os
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.analyze_video import analyze_video

class VideoHandler(FileSystemEventHandler):
    def __init__(self, config_path, output_folder):
        self.config_path = config_path
        self.output_folder = output_folder

    def on_created(self, event):
        # Only process files (not directories) with video extensions.
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(('.mp4', '.avi')):
            return
        
        video_path = event.src_path
        print(f"[INFO] New video file detected: {video_path}")
        
        # Derive output CSV path (using the same base filename, with .csv extension).
        base = os.path.basename(video_path)
        csv_filename = os.path.splitext(base)[0] + '.csv'
        output_csv_path = os.path.join(self.output_folder, csv_filename)
        
        try:
            # Call your video analysis function.
            analyze_video(video_path, self.config_path, output_csv_path)
            print(f"[INFO] Processed video '{video_path}'. Output saved to '{output_csv_path}'.")
        except Exception as e:
            print(f"[ERROR] Error processing video '{video_path}': {e}")
            return
        
        # Delete the video file after processing.
        try:
            os.remove(video_path)
            print(f"[INFO] Deleted video file '{video_path}' after processing.")
        except Exception as e:
            print(f"[WARNING] Failed to delete '{video_path}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Monitor an input folder for new video files, process them, and delete them after analysis."
    )
    parser.add_argument("--input_folder", default="videos/input/", help="Folder to monitor for new video files")
    parser.add_argument("--output_folder", default="videos/output/", help="Folder to store output CSV files")
    parser.add_argument("--config", required=True, help="Path to the DLC config.yaml file")
    args = parser.parse_args()
    
    # Create output folder if it does not exist.
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    event_handler = VideoHandler(args.config, args.output_folder)
    observer = Observer()
    observer.schedule(event_handler, path=args.input_folder, recursive=False)
    observer.start()
    
    print(f"[INFO] Monitoring folder '{args.input_folder}' for new video files...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Stopping folder monitoring...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
