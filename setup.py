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

def load_model_and_tokenizer():
    # Load GPT2-xl from Huggingface.
    model_name = "gpt2-xl"
    print(f"Loading model {model_name}", flush=True)
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    )
    return mt


def setup(model_name="gpt2-xl", mode = "analysis"):
    stopwords0_ = stopwords.words('english')
    stopwords0_ = {word: "" for word in stopwords0_}

    # Get CounterFact data for GPT2-xl, from the ROME repository.
    known_path = "known_1000.json"
    if not os.path.exists(known_path):
        wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
    knowns_df = pd.read_json(known_path)

    # Load GPT2-xl from Huggingface.
    mt = load_model_and_tokenizer()
    mt.model.eval()

    # knowns_df = None
    if mode == "analysis":
        entropy_path = "/home/gamir/DER-Roei/dhgottesman/NLP_FINAL/slurm_dissecting_factual_predictions/enriched_df/false_examples_only_entropy_part_0.json"
        entropy_df = pd.read_json(entropy_path)
        return dict(
        stopwords0_=stopwords0_,
        knowns_df=knowns_df,
        entropy_df=entropy_df,
        mt=mt)
    else: #mode == "dataset_creation"
        # for entropy_df creation. 
        counterfact_path = "counterfact.json"
        if not os.path.exists(counterfact_path):
            wget.download("https://rome.baulab.info/data/dsets/counterfact.json")
        counterfact_df = pd.read_json(counterfact_path)
        return dict(
            stopwords0_=stopwords0_,
            knowns_df=knowns_df,
            counterfact_df=counterfact_df,
            mt=mt
        )