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

from indic_llm import convert_to_corpus