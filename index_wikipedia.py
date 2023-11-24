#!/usr/bin/python3

import argparse
import boto3
import json

from elasticsearch import Elasticsearch, RequestsHttpConnection, TransportError
# from elasticsearch.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
from smart_open import open
from tqdm import tqdm

import re 
from wiki_dump_reader import Cleaner, iterate
import logging
import os 

from aws_utils import MAPPING, get_esclient


os.environ["AWS_PROFILE"] = "daniela"
job_id = os.getenv("SLURM_JOB_ID", "")
logging.basicConfig(filename=f'/home/gamir/DER-Roei/dhgottesman/strategyqa/elasticsearch_index/wikipedia_index_{job_id}.log', encoding='utf-8', level=logging.DEBUG)


# Define cleaning rules as functions
def remove_headers(text):
    return re.sub(r'=+\s*[^=]*\s*=+', '', text)

def remove_newlines(text):
    return text.replace("\n", "")

 # Function that constructs a json body to add each line of the file to index
def make_documents(f):
    processed_docs = 0
    doc_count = 0
    doc_id = 0
    cleaner = Cleaner()
    batch = []
    l_idx = -1
    for l in tqdm(f):
        l_idx += 1
        # if l_idx <= 12500000:
        #     continue
        para_json = json.loads(l)

        # Skip metadata docs and disambiguation pages
        skip = len(para_json.keys()) == 1
        if not skip:
            for category in para_json["category"]:
                if "disambiguation" in category.lower():
                    skip = True
                    break
        if skip:
            continue
        
        # Add auxilary text doc
        secID = 0
        auxilary_text = para_json.get('auxiliary_text', "")
        if auxilary_text:
            metadata = { "index": {"_index": index_name, "_id": f"{doc_id}:{secID}"}}
            content = {
                'docId': para_json.get("page_id", -1),
                'secId': secID,
                'title': para_json.get("title", ""),
                'text': auxilary_text,
                'category': para_json.get("category", ""),
                'createTimestamp': para_json.get("create_timestamp", ""),
                'timestamp': para_json.get("timestamp", ""),
            }
            if processed_docs % 500 == 0:
                print(f"Line {l_idx}")
                logging.info(f"Line {l_idx}")
                try:
                    res = es.bulk(batch, timeout=300)
                    doc_count += len(res["items"])
                    logging.info("Errors {0}. Took {1}. Index {2}. Added {3} documents.".format(res["errors"], res["took"], index_name, doc_count))
                except Exception as e:
                    logging.info(json.dumps(batch))
                batch = []
            batch.append(metadata)
            batch.append(content)
            processed_docs += 2
            secID += 1

        # Split the page into paragraphs and index each paragraph separately
        paragraphs = para_json["source_text"].split("\n\n")
        for paragraph in paragraphs:
            text = cleaner.clean_text(paragraph)
            cleaned_text, _ = cleaner.build_links(text)
            cleaned_text = remove_headers(cleaned_text) 
            cleaned_text = remove_newlines(cleaned_text)
            if cleaned_text == "":
                continue
            metadata = { "index": {"_index": index_name, "_id": f"{doc_id}:{secID}"}}
            content = {
                'docId': para_json.get("page_id", -1),
                'secId': secID,
                'title': para_json.get("title", ""),
                'text': cleaned_text,
                'category': para_json.get("category", ""),
                'createTimestamp': para_json.get("create_timestamp", ""),
                'timestamp': para_json.get("timestamp", ""),
            }
            if processed_docs % 500 == 0:
                print(f"Line {l_idx}")
                logging.info(f"Line {l_idx}")
                try:
                    res = es.bulk(batch, timeout=300)
                    doc_count += len(res["items"])
                    logging.info("Errors {0}. Took {1}. Index {2}. Added {3} documents.".format(res["errors"], res["took"], index_name, doc_count))
                except Exception as e:
                    logging.info(json.dumps(batch))
                batch = []
            batch.append(metadata)
            batch.append(content)
            processed_docs += 2
            secID += 1
        doc_id += 1



if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Add paragraphs from a JSON file to an Elasticsearch index.')
    parser.add_argument('--host', help='Elastic Search hostname')
    parser.add_argument('--port', default=9200, help='Port number')
    parser.add_argument('--region', default=None, help='The region for AWS ES Host. '
                        'If set, we use AWS_PROFILE credentials to connect to the host')
    parser.add_argument('file', help='Path of file to index, e.g. /path/to/my_corpus.json')
    parser.add_argument('index', help='Name of index to create')
    
    args = parser.parse_args()

    # Get Index Name
    index_name = args.index

    # Get an ElasticSearch client
    es = get_esclient(args.host, args.port, args.region)

    # Create an index, ignore if it exists already
    try:
        res = es.indices.create(index=index_name, ignore=400, body=MAPPING)
        print(res)
        # Bulk-insert documents into index
        print(args.file)
        with open(args.file, "r") as f:
            make_documents(f)
    except Exception as inst:
        print(inst)


