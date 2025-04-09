import cv2

def test_codec_availability():
    codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    fps = 30.0
    frame_size = (1280, 720)
    out = cv2.VideoWriter('test_output.mp4', fourcc, fps, frame_size)

    if out.isOpened():
        print(f"Codec {codec} is available and VideoWriter is opened successfully.")
        out.release()
    else:
        print(f"Failed to open VideoWriter with codec {codec}.")

if __name__ == "__main__":
    test_codec_availability()
