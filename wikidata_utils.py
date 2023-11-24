import pandas as pd
import requests
from tqdm import tqdm
import os

def retrieve_property_info(property_id):
    # Define the Wikidata API endpoint
    api_url = 'https://www.wikidata.org/w/api.php'

    # Define parameters for the API request
    params = {
        'action': 'wbgetentities',
        'ids': property_id,
        'format': 'json',
        'languages': 'en',  # Filter for English labels
        'props': ["aliases"]
    }

    # Make the API request
    response = requests.get(api_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        entity = data.get('entities', {}).get(property_id, {})

        # Check if the entity has labels in English
        aliases = []
        if 'aliases' in entity and "en" in entity["aliases"]:
            aliases = [alias["value"] for alias in entity["aliases"]["en"]]
        else:
            print(f'Property ID {property_id} has no aliases.')
        
        return entity["type"], entity["datatype"], aliases
    else:
        print(f'Failed to retrieve data. Status code: {response.status_code}')

def retrieve_all_properties_infos(df):
    ids, types, datatypes, aliases = [], [], [], []
    for _, row in tqdm(df.iterrows()):
        t, d, a = retrieve_property_info(row.id)
        ids.append(row.id)
        types.append(t)
        datatypes.append(d)
        aliases.append(a)
    info_df = pd.DataFrame({"id": ids, "type": types, "datatype": datatypes, "aliases": aliases})
    info_df.to_json("relation_tracing_data/_wikidata_properties_type_aliases.json")
    result = df.merge(info_df, on="id")
    return result

def retrieve_all_properties():
    # Define the Wikidata Query Service API endpoint
    api_url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"

    # Define the SPARQL query
    sparql_query = """
    SELECT ?property ?propertyLabel
    WHERE {
    ?property a wikibase:Property .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    """

    # Define headers for the API request
    headers = {
        "Accept": "application/sparql-results+json",
    }

    # Define the request parameters
    params = {
        "query": sparql_query,
        "format": "json",
    }

    # Make the API request
    response = requests.get(api_url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        ids, labels = [], []
        for item in data["results"]["bindings"]:
            property_id = item["property"]["value"].split("/")[-1]
            label = item["propertyLabel"]["value"]
            ids.append(property_id)
            labels.append(label)
        df = pd.DataFrame({"id": ids, "label": labels})
        df.to_json("relation_tracing_data/wikidata_properties.json")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

def count_relation_types(df):
    df = df.groupby('relation_id').agg(count=('relation_id', 'count')).reset_index()
    df = df.sort_values(by='count', ascending=False).reset_index()
    return df

def load_df_from_dir(directory):
    files = [f for f in os.listdir(directory) if f[-5:] == ".json"]
    df = None
    for f in files:
        fp = os.path.join(directory, f)
        if df is None:
            df = pd.read_json(fp)
        else:
            df = pd.concat([df, pd.read_json(fp)])
    return df.reset_index()



if __name__ == "__main__":
    mistakes_properties_df = pd.read_json("relation_tracing_data/mistakes_wikidata_properties.json")
    knowns_properties_df = pd.read_json("relation_tracing_data/knowns_wikidata_properties.json")
    print("here")
    # df = pd.read_json(wikidata_properties_path)
    # result = retrieve_all_properties_infos(df)
    # result.to_json("relation_tracing_data/combined_wikidata_properties.json")