import os
from PIL import Image

def get_video_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def make_gif_from_segments(video_basename, output_dir="segments", output_gif_dir=".", duration=200):
    files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(f"{video_basename}_frame_") and f.endswith(".jpg")],
        key=lambda x: int(x.split("_frame_")[-1].split(".")[0])
    )

    if not files:
        print(f"No frames found for {video_basename}.")
        return

    frames = [Image.open(os.path.join(output_dir, f)) for f in files]
    gif_path = os.path.join(output_gif_dir, f"{video_basename}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"Saved GIF: {gif_path}")

if __name__ == "__main__":
    VIDEO1 = "kangur1.mp4"
    VIDEO2 = "kangur2.mp4"

    make_gif_from_segments(get_video_basename(VIDEO1))
    make_gif_from_segments(get_video_basename(VIDEO2))
