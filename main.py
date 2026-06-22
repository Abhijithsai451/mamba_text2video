from data_utils.data_import import download_data


def app():

    download_data()
    import torchvision

    # 1. Verify torchvision can see the PyAV backend
    backends = torchvision.io.get_video_backends()
    print(f"Available video backends: {backends}")

    # 2. Try initializing the VideoReader
    try:
        from torchvision.io import VideoReader
        print("Success: VideoReader imported smoothly!")
    except ImportError:
        # If the IDE still complains but 'av' is in the backends list,
        # you can access it dynamically like this:
        if hasattr(torchvision.io, 'VideoReader'):
            print("Success: VideoReader exists dynamically via torchvision.io")
        else:
            print("VideoReader is not exposed. Use torchvision.io.read_video instead.")

if __name__ == "__main__":
    app()