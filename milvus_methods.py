from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import weaviate.classes.config as wc
from weaviate.classes.config import Configure
import os
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure
import time
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from pymilvus import MilvusClient

IMAGE_PATH = "DATA/jpeg_images/"

def to_string(row):
    return ", ".join([
        str(row.get('color', '')).strip(),
        str(row.get('craft', '')).strip(),
        str(row.get('fabric', '')).strip(),
        str(row.get('Product Types', '')).strip()
    ])

def input_to_embeddings(user_query, model):
    return model.encode(user_query)

def best_results(image_description_result, description_result, tags_result, model, query, client):
    all_handle = set()
    results_map= defaultdict(dict)

    description_result = description_result[0]
    image_description_result = image_description_result[0]
    tags_result = tags_result[0]

    for obj in description_result:
        handle = obj.entity.get('Handle')
        if handle:
            results_map[handle]['description_vector'] = obj.get('distance')
            all_handle.add(obj.get('Handle'))
    for obj in tags_result:
        handle = obj.entity.get('Handle')
        if handle:
            results_map[handle]['tags_vector'] = obj.get('distance')
            all_handle.add(obj.get('Handle'))
    for obj in image_description_result:
        handle = obj.entity.get('Handle')
        if handle:
            results_map[handle]['image_description_vector'] = obj.get('distance')
            all_handle.add(obj.get('Handle'))

    query_vector = model.encode(query)
    
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1) 
    final_result = []
    
    for handle in all_handle: 
        dist_dict = {}

        res = client.query(collection_name="SemanticSearch",
                           filter=f'Handle == "{handle}"',
                           output_fields=["Handle", "Title", "Type", "description", "Description", "Tags", 'craft', 'fabric', 'color', "Product_Types", "description_vector", "tags_vector", "image_description_vector"]
                           )
        obj = res[0]

        if not obj:
            continue

        vectors = {}
        vectors['description_vector'] = obj.get('description_vector')
        vectors['image_description_vector'] = obj.get('image_description_vector')
        vectors['tags_vector'] = obj.get('tags_vector')

        for vec_name in ['description_vector', 'image_description_vector', 'tags_vector']:
            if vec_name in vectors:
                product_vec = np.array(vectors[vec_name])
                distance = 1 - cosine_similarity(query_vector, [product_vec])[0][0]
                dist_dict[vec_name] = distance
        
        distances = list(dist_dict.values())
        avg_distance = sum(distances) / len(distances)
        result_entry = {"handle": handle,
                        "distance": avg_distance
                        }

        for vec_name in ['description_vector', 'image_description_vector', 'tags_vector']:
            if vec_name in dist_dict:
                result_entry[vec_name] = dist_dict[vec_name]
                result_entry['description'] = obj.get('description')
                result_entry['Description'] = obj.get('Description')
                result_entry['fabric'] = obj.get('fabric')
                result_entry['craft'] = obj.get('craft')
                result_entry['color'] = obj.get('color')
                result_entry['Product_Types'] = obj.get('Product_Types')

        final_result.append(result_entry)
    final_result.sort(key=lambda x: x["distance"])

    return final_result

def similarity_search_VDB(user_query, model, client, top_k=5):
   
    query_vector = model.encode(user_query)

    description_result = client.search(
                                collection_name="SemanticSearch",
                                anns_field="description_vector",
                                data=[query_vector],
                                limit=top_k,
                                search_params={"metric_type": "COSINE", "params": {}},
    )
    
    image_description_result = client.search(
                                collection_name="SemanticSearch",
                                anns_field="image_description_vector",
                                data=[query_vector],
                                limit=top_k,
                                search_params={"metric_type": "COSINE", "params": {}},
    )

    tags_result = client.search(
                                collection_name="SemanticSearch",
                                anns_field="tags_vector",
                                data=[query_vector],
                                limit=top_k,
                                search_params={"metric_type": "COSINE", "params": {}},
                                # output_fields=["Handle", "Title"]
)
    combined_results = best_results(description_result,
                                    image_description_result,
                                    tags_result,
                                    model,
                                    user_query,
                                    client
                                    )
    # print("description_result", type(description_result))
    return combined_results

def search_products(handles, df):
    products = []

    for handle in handles:
        if isinstance(handle, str) and handle.strip() != "":
            product_data = df.loc[df['Handle'] == handle, [
                "Handle", "Title", "Type", "description", 
                "fabric", "Product Types", "craft", "Description", "color"
            ]]

            if product_data.empty:
                continue

            # Load image safely
            image_path = os.path.join(IMAGE_PATH, f"{handle}.jpg")
            image = None
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Error loading image for {handle}: {e}")

            products.append({
                "handle": handle,
                "data": product_data.iloc[0],  # Series
                "image": image
            })

    return products



