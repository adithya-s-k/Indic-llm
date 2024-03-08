import logging
import os
import json

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download(
    dataset_name,
    dataset_subset,
    dataset_split="train",
    ) -> Dataset:
    
    dataset = load_dataset(
        dataset_name,
        dataset_subset,
        split=dataset_split,
        keep_in_memory=True
    )
    
    return dataset

def download_multiple() -> None:
    pass

def convert_to_corpus(
    dataset_name,
    dataset_subset,
    dataset_split,
    text_col,
    output_file_name,
    output_dir = "./corpus"
    ) -> None:
    
    try:
        dataset = load_dataset(dataset_name,dataset_subset,split=dataset_split)
        train_df = pd.DataFrame(dataset)

        os.makedirs(output_dir, exist_ok=True)
        corpus_path = os.path.join(output_dir, output_file_name)

        with open(corpus_path, "w") as file:
            for index, value in tqdm(
                train_df[text_col].items(), total=len(train_df)
            ):
                file.write(str(value) + "\n")

    except Exception as e:
        logger.error(f"Error creating the text corpus -> {e}")

    return corpus_path


def truncate_text_corpus() -> None:
    pass

