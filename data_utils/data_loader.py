import os
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
tokenizer_name = os.getenv("TOKENIZER")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


def download_data():
    if not os.path.exists(dataset_path):
        dataset = download_data
    else:
        dataset = load_from_disk(dataset_path)

def cinepile_collate_fn(batch):
    """
    Custom collate function to handle the required fields in the dataset
    """
    video_ids = [item['video_id'] for item in batch]
    movie_names = [item['movie_name'] for item in batch]
    genres = [item['genres'] for item in batch]

    labels = torch.tensor([int(item['answer_key_position']) for item in batch], dtype=torch.long)
    questions = [item['question'] for item in batch]
    subtitles = [item['subtitles'] for item in batch]

    tokenized_text = tokenizer(
        questions,
        text_pair=subtitles,  # Optionally pair the question with subtitles as context
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    all_choices_ids = []
    all_choices_masks = []

    for item in batch:
        tokenized_choices = tokenizer(
            item['choices'],
            padding='max_length',
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )
        all_choices_ids.append(tokenized_choices['input_ids'])  # Shape: [num_choices, seq_len]
        all_choices_masks.append(tokenized_choices['attention_mask'])  # Shape: [num_choices, seq_len]

    # Stack individual samples into uniform batch tensors
    # Resulting shape: [batch_size, num_choices, seq_len]
    choices_input_ids = torch.stack(all_choices_ids)
    choices_attention_mask = torch.stack(all_choices_masks)

    # --- Construct Final Batch Dictionary ---
    return {
        # Metadata
        "video_ids": video_ids,
        "movie_names": movie_names,
        "genres": genres,

        # Primary Model Inputs (Tensors)
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],

        # Choice Inputs (Tensors)
        "choices_input_ids": choices_input_ids,
        "choices_attention_mask": choices_attention_mask,

        # Target Label (Tensor)
        "labels": labels
    }


def load_video_data(dataset: Dataset, ):
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 4,
        shuffle = True,
        collate_fn = cinepile_collate_fn,
        num_workers = 4,
        pin_memory = True
    )




