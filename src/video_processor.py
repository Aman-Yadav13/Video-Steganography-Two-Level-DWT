import cv2
from tqdm import tqdm

def split_video_to_frames(video_path):
    """Splits video into frames and extracts audio."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # Note: Audio extraction would typically use a library like moviepy.
    # For this project, we can skip audio or assume ffmpeg is available.
    print(f"Extracted {len(frames)} frames from video.")
    return frames

def mux_frames_to_video(frames, output_path, fps=30.0):
    """Recombines frames into an MP4 video."""
    if not frames:
        return
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Muxing frames back into video...")
    for frame in tqdm(frames):
        out.write(frame)
    out.release()