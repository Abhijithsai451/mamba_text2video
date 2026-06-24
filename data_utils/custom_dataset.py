import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import torchvision.io as io
import torchvision.transforms.v2 as transforms


def cinepile_dataset(Dataset):
    def __init__(self, parquet_path, video_dir, num_frames = 32, max_duration= 10,height=224, width=224 ):
        print("Loading CinePile metadata sheet...")
        self.df = pd.read_parquet(parquet_path)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.max_duration = max_duration

        # Apple Silicon Optimized Native Transforms (V2 framework)
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToDtype(torch.float32, scale=True),  # Scaled mapping: [0, 255] uint8 -> [0.0, 1.0] float32
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet distribution
        ])

        print(f"Dataset successfully initialized with {len(self.df)} metadata rows.")

    def __len__(self):
        return len(self.df)

    def _calculate_uniform_indices(self, total_frames):
        """Generates perfectly equidistant integers across the parsed slice window."""
        if total_frames <= self.num_frames:
            return list(range(total_frames))

        # Linspace generates float indices, which we cast to strict sequential integers
        indices = torch.linspace(0, total_frames - 1, steps=self.num_frames)
        return [int(idx.item()) for idx in indices]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_id = row.get('video_id', row.get('yt_clip_link', str(idx))).split('=')[-1]
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)

        if not os.path.exists(video_path):
            return self.__getitem__((idx + 1) % len(self))

        try:
            reader = io.VideoReader(video_path, "video")
            metadata = reader.get_metadata()

            fps = metadata['video'].get('fps', [30])[0]
            actual_duration = metadata['video'].get('duration', [10.0])[0]

            clamped_duration = min(actual_duration, self.max_duration)
            total_frames_to_sample = int(clamped_duration * fps)

            target_indices = self._calculate_uniform_indices(total_frames_to_sample)
            frames_list = []

            for frame_idx in target_indices:
                pts_timestamp = frame_idx / fps
                reader.seek(pts_timestamp)
                try:
                    frame_data = next(reader)
                    frames_list.append(frame_data['data'])
                except StopIteration:
                    break

            if len(frames_list) == 0:
                frames_list = [torch.zeros((3, 224, 224), dtype=torch.uint8)]

            video_tensor = torch.stack(frames_list)
            video_tensor = self.transform(video_tensor)

        except Exception as video_read_error:
            print(f"Warning: Issue parsing index {idx} ({video_filename}): {video_read_error}. Skipping...")
            return self.__getitem__((idx + 1) % len(self))

        question_text = str(row.get('question', ''))
        answer_key = str(row.get('answer_key', ''))  # e.g. "A", "B", "C", "D", "E"

        return {
            "video": video_tensor,
            "question": question_text,
            "answer": answer_key
        }