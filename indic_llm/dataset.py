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


def download_dataset(
    dataset_name,
    dataset_subset,
    dataset_split="train",
    ) -> None:
    
    logger.info("Starting to download/load dataset")
    
    try:
        load_dataset(
            dataset_name,
            dataset_subset,
            split=dataset_split,
            keep_in_memory=True
        )
        
        logger.info("Downloading dataset completed successfully")
    except Exception as e:
        logger.error("An error occurred while downloading dataset")
        logger.error(f"Exception: {e}")

def download_multiple() -> None:
    pass

def download_convert_to_txt(
    dataset_name,
    dataset_subset,
    dataset_split,
    text_col,
    output_file_name,
    output_dir = "./corpus"
    ) -> None:
    
    try:
        logger.info("Starting to download/load dataset")
        dataset = load_dataset(dataset_name,dataset_subset,split=dataset_split)
        logger.info("Dataset loaded/downloaded successfully")
        logger.info("Converting to pandas dataframe")
        train_df = pd.DataFrame(dataset)
        logger.info("Conversion complete")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            corpus_path = os.path.join(output_dir, output_file_name)
            logger.info(f"Created output directory: {output_dir}")
        else:
            logger.info(f"Output directory already exists: {output_dir}")

        logger.info("Creating Text corpus")
        with open(corpus_path, "w") as file:
            for index, value in tqdm(
                train_df[text_col].items(), total=len(train_df)
            ):
                file.write(str(value) + "\n")
        logger.info("CText corpus Created Successfully")

    except Exception as e:
        logger.error(f"Error creating the text corpus -> {e}")

    return corpus_path


def truncate_text_corpus() -> None:
    pass

