import sys
import os
import json
import shutil
import logging
import argparse
from tqdm import tqdm


import pandas as pd
from datasets import load_dataset
from dataclasses import asdict
from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from indic_llm.dataset import download_dataset, download_convert_to_txt
from indic_llm import print_indic_llm_text_art

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
        parser.add_argument(
            "--generate-corpus",
            type=bool, 
            required=False,
            default = False,
            help="Generate text corpus from the dataset"
        )
        parser.add_argument(
            "--text-column",
            type=str,
            required=False,
            default="text",
            help="the text column of the dataset to concatenate and create the text corpus"
        )
        parser.add_argument(
            "--output-file-name",
            type=str,
            required=False,
            default="text",
            help="name of the output corpus text file formate: {input_file_name}.txt"
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            required=False,
            default="./corpus",
            help="name of the output directory where you want to save text corpus"
        )

        args = parser.parse_args()
        
        # Initialize logger with appropriate log level
                # Log parsed arguments
        logger.info("Parsed arguments:")
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")

        print_indic_llm_text_art()

        logger.setLevel(logging.INFO)
        # if generate corpus is true
        # verify is generate_corpus is boolean (default: False)
        # assert args.generate_corpus is bool, "--generate-corpus should be True or False"
        if args.generate_corpus == True:
            assert args.text_column != "", "Text column must not be empty"
            assert args.output_file_name != "", "Output file name must not be empty"
            assert args.output_file_name.endswith(".txt"), "Output file name should end with '.txt'"
            
            download_convert_to_txt(
                args.hf_dataset,
                args.hf_subset,
                args.dataset_split,
                args.text_column,
                args.output_file_name,
                args.output_dir
            )
        # if generate corpus is false
        elif args.generate_corpus == False:
            download_dataset(
                args.hf_dataset,
                args.hf_subset,
                args.dataset_split
            )
        else:
            logger.error("Invalid input for --generate_corpus. Use 'True' or 'False'.")
        
if __name__ == "__main__":
    download_dataset = DownloadDataset()
    download_dataset.run()