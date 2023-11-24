import os
import wget

# Scientific packages
import pandas as pd

# Utilities
from utils import ModelAndTokenizer

# List of stopwords from NLTK, needed only for the attributes rate evaluation.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def setup(model_name="gpt2-xl"):
    stopwords0_ = stopwords.words('english')
    stopwords0_ = {word: "" for word in stopwords0_}

    # Get CounterFact data for GPT2-xl, from the ROME repository.
    known_path = "known_1000.json"
    if not os.path.exists(known_path):
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json(known_path)
    # knowns_df = None
    entropy_path = "/home/gamir/DER-Roei/dhgottesman/NLP_FINAL/slurm_dissecting_factual_predictions/enriched_df/false_examples_only_entropy_part_0.json"
    entropy_df = pd.read_json(entropy_path)

    # Load GPT2-xl from Huggingface.
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    )
    mt.model.eval()
    return dict(
        stopwords0_=stopwords0_,
        knowns_df=knowns_df,
        entropy_df=entropy_df,
        mt=mt
    )