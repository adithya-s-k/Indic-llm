import argparse
import logging
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from indic_llm.dataset import download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DownloadDataset():
    def __init__(self) -> None:
        pass
    
    def run(self):
        parser = argparse.ArgumentParser(
            description="Download Hugging Face dataset."
        )
        parser.add_argument(
            "--hf-dataset",
            required=True,
            help="Name of the Hugging Face dataset (e.g., 'imdb').",
        )
        parser.add_argument(
            "--hf-subset",
            required=False,
            help="Name of the path to language inside the dataset"
        )
        parser.add_argument(
            "--dataset-split",
            default="train",
            help="Dataset split to use (default: 'train').",
        )
        

        args = parser.parse_args()
        download(
            args.hf_dataset,
            args.hf_subset,
            args.dataset_split
        )
        
if __name__ == "__main__":
    download_dataset = DownloadDataset()
    download_dataset.run()