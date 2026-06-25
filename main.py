import os
from torch.utils.data import DataLoader
from data_utils.custom_dataset import CinepileDataset
from data_utils.data_loader import load_video_data
from dotenv import load_dotenv

load_dotenv()

def app():
    print("\nTesting DataLoader iteration...")

    dataloader = load_video_data()
    data_dir  = os.getenv("DATASET_PATH")

    v1_dir = "./v1"  # Or wherever your downloaded MP4 directory is located

    try:
        cinepile_dataset = CinepileDataset(
            dataset_dir=data_dir,
            video_dir=LOCAL_VIDEO_DIRECTORY,
            num_frames=16,
            max_duration=10.0
        )

        test_loader = DataLoader(cinepile_dataset, batch_size=1, shuffle=True)

        print("\nStarting execution check...")
        for step, batch_item in enumerate(test_loader):
            print(f"\n--- Batch Step {step + 1} Extracted Successfully ---")
            print(f"Video Tensor Data Layout Shape (B, T, C, H, W): {batch_item['video'].shape}")
            print(f"Parsed Question Payload: {batch_item['question'][0]}")
            print(f"Ground Truth Answer ID Target: {batch_item['answer'][0]}")
            break

    except Exception as e:
        print(f"\nExecution failed. Ensure paths align with your directory setup.\nDetails: {e}")
if __name__ == "__main__":
    app()