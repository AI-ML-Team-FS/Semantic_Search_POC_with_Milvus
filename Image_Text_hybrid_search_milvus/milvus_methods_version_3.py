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
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
import pandas as pd
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
from pymilvus import Function, FunctionType
import streamlit
IMAGE_PATH = "DATA/jpeg_images/"

df = pd.read_csv('DATA/products_with_tags_and_description_seperated.csv')

# collection_name = "SemanticSearch"
collection_name = "SemanticSearch_bgremoved"

def get_image_vector(image_path, model, processor):
    try:
        image = Image.open(image_path).convert("RGB")  # convert ensures it's a valid RGB image
    except (FileNotFoundError, UnidentifiedImageError) as e:
        raise ValueError(f"Invalid image file at {image_path}: {e}")

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    return image_embedding.squeeze().cpu().tolist()

def get_text_vector(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True) 

    return text_embedding.squeeze().cpu().tolist()
def to_string(row):
    return ", ".join([
        str(row.get('color', '')).strip(),
        str(row.get('craft', '')).strip(),
        str(row.get('fabric', '')).strip(),
        str(row.get('Product Types', '')).strip()
    ])
def hybrid_search_image_and_text(user_query, user_image, client, model, processor, top_k=5):
    query_vector = get_text_vector(user_query, model, processor)    
    image_query_vector = get_image_vector(user_image, model, processor)

    search_param_1 = {
        "data": [query_vector],
        "anns_field": "description_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    search_param_2 = {
        "data": [query_vector],
        "anns_field": "image_description_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    search_param_3 = {
        "data": [query_vector],
        "anns_field": "tags_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }

    search_param_4 = {
        "data": [user_query],
        "anns_field": "description_vector_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": 10
    }
    search_param_5 = {
        "data": [user_query],
        "anns_field": "image_description_vector_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": 10
    }
    search_param_6 = {
        "data": [user_query],
        "anns_field": "tags_vector_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": 10
    }
    search_param_7 = {
        "data": [query_vector],
        "anns_field": "image_dense",
        "param": {"nprobe": 10},
        "limit": 10
    }
    text_request_1 = AnnSearchRequest(**search_param_1)
    text_request_2 = AnnSearchRequest(**search_param_2)
    text_request_3 = AnnSearchRequest(**search_param_3)
    text_request_4 = AnnSearchRequest(**search_param_4)
    text_request_5 = AnnSearchRequest(**search_param_5)
    text_request_6 = AnnSearchRequest(**search_param_6)
    text_request_7 = AnnSearchRequest(**search_param_7)

    search_param_1 = {
    "data": [image_query_vector],
    "anns_field": "description_vector",
    "param": {"nprobe": 10},
    "limit": 10
    }
    search_param_2 = {
        "data": [image_query_vector],
        "anns_field": "image_description_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    search_param_3 = {
        "data": [image_query_vector],
        "anns_field": "tags_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }

    search_param_4 = {
        "data": [image_query_vector],
        "anns_field": "image_dense",
        "param": {"nprobe": 10},
        "limit": 10
    }

    image_request_1 = AnnSearchRequest(**search_param_1)
    image_request_2 = AnnSearchRequest(**search_param_2)
    image_request_3 = AnnSearchRequest(**search_param_3)
    image_request_4 = AnnSearchRequest(**search_param_4)

    text_search_requests = [text_request_1, text_request_2, text_request_3, text_request_4, text_request_5, text_request_6, text_request_7]
    image_search_requests = [image_request_1, image_request_2, image_request_3, image_request_4]
    
    # ranker = RRFRanker(100)
    ranker = Function(
    name="weight",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "weighted", 
        "weights": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1 ,0.35],
        # "norm_score": True  # Optional
            }
    )

    results = client.hybrid_search(
        collection_name = collection_name,
        reqs = [text_request_1,text_request_2,text_request_3,text_request_4,text_request_5,text_request_6,text_request_7,image_request_1,image_request_2,image_request_3,image_request_4],
        ranker = ranker,
        limit = int(top_k)
    )
    products=[]
    for obj in results[0]:
        handle = obj.get('Handle')
        distance = obj.get('distance')
        products.append({
            "handle": handle,
            "distance": distance
            })

    return products

def hybrid_search_image_only(user_image, client, model, processor, top_k=5):  
    image_query_vector = get_image_vector(user_image, model, processor)
    # st.write(image_query_vector)

    search_param_1 = {
        "data": [image_query_vector],
        "anns_field": "description_vector",
        "param": {"nprobe": 10},
        "limit": 10
        }
    search_param_2 = {
        "data": [image_query_vector],
        "anns_field": "image_description_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    search_param_3 = {
        "data": [image_query_vector],
        "anns_field": "tags_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    search_param_4 = {
        "data": [image_query_vector],
        "anns_field": "image_dense",
        "param": {"nprobe": 10},
        "limit": 10
    }

    image_request_1 = AnnSearchRequest(**search_param_1)
    image_request_2 = AnnSearchRequest(**search_param_2)
    image_request_3 = AnnSearchRequest(**search_param_3)
    image_request_4 = AnnSearchRequest(**search_param_4)

    image_search_requests = [image_request_1, image_request_2, image_request_3, image_request_4]
    
    # ranker = RRFRanker(100)
    ranker = Function(
    name="weight",
    input_field_names=[], # Must be an empty list
    function_type=FunctionType.RERANK,
    params={
        "reranker": "weighted", 
        "weights": [0.1, 0.1, 0.1, 0.7],
        "norm_score": True  # Optional
    }
    )
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[image_request_1,image_request_2,image_request_3,image_request_4],
        ranker=ranker,
        limit=int(top_k)
    )
    products=[]
    for obj in results[0]:
        handle = obj.get('Handle')
        distance = obj.get('distance')
        products.append({
            "handle": handle,
            "distance": distance
            })
    return products

def hybrd_search_text_only(user_query, model, processor, client, top_k=5):

    query_vector = get_text_vector(user_query, model, processor)
    # searching over dense vectors  
    search_param_1 = {
    "data": [query_vector],
    "anns_field": "description_vector",
    "param": {"nprobe": 10},
    "limit": 10
    }
    search_param_2 = {
        "data": [query_vector],
        "anns_field": "image_description_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    search_param_3 = {
        "data": [query_vector],
        "anns_field": "tags_vector",
        "param": {"nprobe": 10},
        "limit": 10
    }
    request_1 = AnnSearchRequest(**search_param_1)
    request_2 = AnnSearchRequest(**search_param_2)
    request_3 = AnnSearchRequest(**search_param_3)

    search_param_4 = {
        "data": [user_query],
        "anns_field": "description_vector_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": 10
    }
    search_param_5 = {
        "data": [user_query],
        "anns_field": "image_description_vector_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": 10
    }
    search_param_6 = {
        "data": [user_query],
        "anns_field": "tags_vector_sparse",
        "param": {"drop_ratio_search": 0.2},
        "limit": 10
    }
    # searching over image vectors
    search_param_7 = {
        "data": [query_vector],
        "anns_field": "image_dense",
        "param": {"nprobe": 10},
        "limit": 10
    }
    
    request_4 = AnnSearchRequest(**search_param_4)
    request_5 = AnnSearchRequest(**search_param_5)
    request_6 = AnnSearchRequest(**search_param_6)
    request_7 = AnnSearchRequest(**search_param_7)

    reqs = [request_1, request_2, request_3, request_4, request_5, request_6, request_7]
    
    ranker = RRFRanker(100)

    res = client.hybrid_search(collection_name=collection_name,
                               reqs=reqs,
                               ranker=ranker,
                               limit=top_k
    )
    products=[]
    for obj in res[0]:
        handle = obj.get('Handle')
        distance = obj.get('distance')
        products.append({
            "handle": handle,
            "distance": distance
            })

    return products

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
                "data": product_data.iloc[0], # Series
                "image": image
            })

    return products