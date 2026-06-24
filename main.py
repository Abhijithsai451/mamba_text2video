from data_utils.data_import import download_data
from data_utils.data_loader import load_video_data


def app():
    print("\nTesting DataLoader iteration...")

    # Grab just the first batch to verify structure
    dataloader = load_video_data()
    for i, batch in enumerate(dataloader):
        print("\n--- Batch Structure Successfully Generated ---")
        print(f"Batch index: {i}")
        print(f"Metadata (Video IDs): {batch['video_ids']}")
        print(f"Main Input IDs Shape: {batch['input_ids'].shape} (Padded uniformly!)")
        print(f"Choices Input IDs Shape: {batch['choices_input_ids'].shape} [Batch, Choices, Seq_len]")
        print(f"Labels Tensor: {batch['labels']}")
        break

if __name__ == "__main__":
    app()