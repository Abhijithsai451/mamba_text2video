from torchvision.io import VideoReader
from dotenv import load_dotenv
import os


load_dotenv()

reader = VideoReader(os.getenv("VIDEO_PATH"))