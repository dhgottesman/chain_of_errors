import boto3

from elasticsearch import RequestsHttpConnection
# from elasticsearch.helpers import bulk
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


MAPPING = {
        "settings": {
            "number_of_shards": 5,
            "index" : {
                "similarity" : {
                    "default" : {
                        "type" : "BM25"
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "false",
            "properties": {
                "docId": { "type" : "integer" },
                "title": { "type" : "text" },
                "heading": { "type" : "text" },
                "auxText": { "type" : "text" },
                "openText": { "type" : "text" },
                "text": { "type" : "text" },
                "category": { "type" : "text" },
                "createTimestamp": { "type" : "date" },
                "timestamp": { "type" : "date" },
            
            }
        }
    }

def get_esclient(host, port, region=None):
    service= 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service)
    return OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        pool_maxsize = 20,
        timeout=30,
        max_retries=10,
        retry_on_timeout=True
    )

