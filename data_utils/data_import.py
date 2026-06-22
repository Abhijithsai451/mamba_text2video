import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
def download_data():
    dataset = load_dataset("tomg-group-umd/cinepile", "v2",
                           token =os.getenv("HF_TOKEN")
                           )
    dataset.save_to_disk("data/")
