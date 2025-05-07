import os
from PIL import Image

def extract_frame_number(filename):
    try:
        return int(filename.split("_frame")[-1].split(".")[0])
    except:
        return -1

def create_gif_from_frames(frame_paths, output_path, fps=2):
    frames = []
    for file in frame_paths:
        frame = Image.open(file)
        frames.append(frame.copy())

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"GIF saved to {output_path}")
    else:
        print("No frames found in folder.")

if __name__ == "__main__":
    input_folder = "highlighted_frames"
    video_groups = {}
    for f in os.listdir(input_folder):
        if f.endswith(".jpg") or f.endswith(".png"):
            key = f.split("_frame")[0]
            video_groups.setdefault(key, []).append(f)

    for video_name, frame_files in video_groups.items():
        sorted_frames = sorted(frame_files, key=extract_frame_number)
        output_gif = f"{video_name}.gif"
        frame_paths = [os.path.join(input_folder, f) for f in sorted_frames]
        create_gif_from_frames(frame_paths, output_gif, fps=2)
