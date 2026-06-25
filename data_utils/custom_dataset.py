import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torchvision.io as io


class CinepileDataset(Dataset):
    def __init__(self, dataset_dir, video_dir, num_frames=32, max_duration = 10.0,height = 224, width = 224 ):
        """
        Args:
            dataset_dir (str): Path to the folder containing 'dataset_dict.json' (root 'data' or specific split).
            video_dir (str): Folder containing your downloaded local .mp4 files.
            num_frames (int): Target count of uniform frames to extract (T).
            max_duration (float): Maximum allowed video context window in seconds.
            height, width (int): Structural frame image dimensions.
        """

        dataset = load_from_disk(dataset_dir)

        if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
            self.dataset = dataset['train']
        else:
            self.dataset = dataset

        self.video_dir = video_dir
        self.num_frames = num_frames
        self.max_duration = max_duration
        self.height = height
        self.width = width

        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToDtype(torch.float32, scale=True),  # Mapping: [0, 255] -> [0.0, 1.0]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Dataset successfully loaded with {len(self.dataset)} metadata entries.")

        def __len__(self):
            return len(self.dataset)

        def _calculate_uniform_indices(self, total_frames):
            """Generates perfectly equidistant integers across the parsed slice window."""
            if total_frames <= self.num_frames:
                return list(range(total_frames))
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames)
            return [int(idx.item()) for idx in indices]

        def __getitem__(self, idx):
            row = self.dataset[idx]
            # Resolve target video file name
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

                # Enforce 10-second context clamp rule
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

                # Standard CinePile schema keywords inside Arrow features
            question_text = str(row.get('question', ''))
            answer_key = str(row.get('answer_key', ''))

            return {
                "video": video_tensor,  # Shape: [num_frames, 3, 224, 224]
                "question": question_text,
                "answer": answer_key
            }



