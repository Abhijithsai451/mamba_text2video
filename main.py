from data_utils.download_data import download
from data_utils.data_loader import load_video_data


def app():
    print("\nTesting DataLoader iteration...")

    dataloader = load_video_data()
    TARGET_FRAMES = 16
    CLAMP_DURATION = 10.0

    if not os.path.exists(LOCAL_PARQUET_FILE):
        print(
            "[!] ERROR: Please update 'LOCAL_PARQUET_FILE' and 'LOCAL_VIDEO_DIRECTORY' with your actual computer strings.")
    else:
        # Construct the pipeline object
        cinepile_dataset = CinePileDataset(
            parquet_path=LOCAL_PARQUET_FILE,
            video_dir=LOCAL_VIDEO_DIRECTORY,
            num_frames=TARGET_FRAMES,
            max_duration=CLAMP_DURATION
        )

        # Instantiate standard DataLoader
        test_loader = DataLoader(cinepile_dataset, batch_size=1, shuffle=True)

        print("\nStarting dry-run batch fetch tracking loop...")
        for step, batch_item in enumerate(test_loader):
            print(f"\n--- Batch Step {step + 1} Extracted Successfully ---")
            print(f"Video Tensor Data Layout Shape (B, T, C, H, W): {batch_item['video'].shape}")
            print(f"Parsed Question Payload: {batch_item['question'][0]}")
            print(f"Ground Truth Answer ID Target: {batch_item['answer'][0]}")

            # Break early after processing 2 validation samples
            if step >= 1:
                print("\n[✔] Phase 1.2 Pipeline verified as computationally clean and active.")
                break
if __name__ == "__main__":
    app()