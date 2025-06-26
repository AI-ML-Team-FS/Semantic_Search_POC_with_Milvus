import streamlit as st
import pandas as pd
# import weaviate
# import weaviate.classes.config as wc
# from weaviate.classes.config import Configure
# import weaviate.classes as wvc
# from weaviate.classes.config import Property, DataType
# from weaviate.classes.config import Configure
from PIL import Image
import time
import os
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager
import milvus_methods
import subprocess
import os
import asyncio
from huggingface_hub import snapshot_download
from pymilvus import MilvusClient, DataType

os.environ["STREAMLIT_WATCH_DIRS"] = ""

MODEL_NAME = 'WhereIsAI/UAE-Large-V1'
path = snapshot_download(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

df = pd.read_csv('DATA/products_with_tags_and_description_seperated.csv')


client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

if not client.has_collection(collection_name="SemanticSearch"):
    # client.drop_collection(collection_name="SemanticSearch")
    
    st.write("Creating collection SemanticSearch")

    schema = MilvusClient.create_schema(auto_id=False)

    schema.add_field(field_name="Handle", datatype=DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field(field_name="Title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="Type", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='craft', datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="fabric", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="Product_Types", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="Tags", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="Description", datatype=DataType.VARCHAR, max_length=1024)

    schema.add_field(field_name="image_description_vector", datatype=DataType.FLOAT_VECTOR, dim = 1024)
    schema.add_field(field_name="description_vector", datatype=DataType.FLOAT_VECTOR, dim = 1024)
    schema.add_field(field_name="tags_vector", datatype=DataType.FLOAT_VECTOR, dim = 1024)

    index_params = MilvusClient.prepare_index_params()

    index_params.add_index(
        field_name="description_vector",
        index_type="FLAT", 
        index_name="description_vector_index",
        metric_type="COSINE", 
        params={} 
    )
    index_params.add_index(
        field_name="image_description_vector", 
        index_type="FLAT",
        index_name="image_description_vector_index",
        metric_type="COSINE",
        params={} 
    )
    index_params.add_index(
        field_name="tags_vector", # Name of the vector field to be indexed
        index_type="FLAT", # Type of the index to create
        index_name="tags_vector_index", # Name of the index to create
        metric_type="COSINE", # Metric type used to measure similarity
        params={} # No additional parameters required for FLAT
    )

    client.create_collection(
        collection_name="SemanticSearch",
        schema=schema,
        index_params=index_params
    )
    res = client.get_load_state(
    collection_name="SemanticSearch"
    )

    embeddings_description = model.encode(df['description'])
    df['Description'] = df['Description'].fillna('')
    embeddings_image_description = model.encode(df['Description'])
    df['image_description_vector'] = embeddings_image_description.tolist()
    df['description_vector'] = embeddings_description.tolist()
    df['Tags'] = df.apply(milvus_methods.to_string, axis=1)
    embeddings_Tags = model.encode(df['Tags'])
    df['Tags_vector'] = embeddings_Tags.tolist()
    df_copy = df.copy()
    df_copy.drop(columns=['Unnamed: 0'], inplace = True)
    df_copy.rename(columns={'Tags_vector': 'tags_vector'}, inplace=True)
    df_copy.rename(columns={'Product Types': 'Product_Types'}, inplace=True)
    df_copy['craft'] = df_copy['craft'].fillna("Unknown").astype(str)
    data = df_copy.to_dict(orient='records')

    res = client.insert(
        collection_name="SemanticSearch",
        data=data
    )    
    
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

st.title("Semantic Search v2: Sue Mue POC")
user_query = st.text_input("Enter your query: ")

if st.button("Search"):

    with st.spinner("Searching for products..."):
        st.write("Searching for products...")
            
        # products_vectordb = client.collections.get("SemanticSearch2")
        
        TOP_K = 5

        handle_score_pairs = milvus_methods.similarity_search_VDB(user_query, model, 
        client, top_k=int(TOP_K))

        # st.write("Search Results from Weaviate: ", handle_score_pairs)
        # st.write("Search Results:")

        handles = [entry["handle"] for entry in handle_score_pairs]
        distances = [entry["distance"] for entry in handle_score_pairs]
        description_vector_distances = [entry["description_vector"] for entry in handle_score_pairs]
        image_description_vector_distances = [entry["image_description_vector"] for entry in handle_score_pairs]
        tags_vector_distances = [entry["tags_vector"] for entry in handle_score_pairs]
        scores = [round(1 - distance, 4)*100 for distance in distances]
        # st.write("Search Results:")

        products = milvus_methods.search_products(handles, df)
        # st.write("Search Results:")
        # st.write("Search Results:", products)

        for i, product in enumerate(products):
            # st.write("INSIDE LOOP")

            data = product["data"]
            image = product["image"]
            distance = distances[i]
            score = scores[i]
            description_vector_distance = description_vector_distances[i]
            image_description_vector_distance = image_description_vector_distances[i]
            tags_vector_distance = tags_vector_distances[i]
            # distance = round(distance, 4)  # Higher = more similar

            st.markdown(f"### {data['Title']}")


            with st.container():
                col1, col2 = st.columns([1, 1])

                with col1:
                    if image:
                        st.image(image)
                    else:
                        st.warning("Image not available.")

                with col2:
                    st.markdown(f"""
                        **Offcial Description:** {data['description']}
                        
                        **Image Description by LLM:** {data['Description']}

                        **Product Type:** {data['Product Types']}

                        **Colors:** {data['color']}

                        **Fabric:** {data['fabric']}  

                        **Craft:** {data['craft']}  

                        **Distance:** {distance}

                        **Score:** {score}

                        | Vector Type                    | Distance |
                        |-------------------------------|----------|
                        | Description Distance   | {round(description_vector_distance, 6)} |
                        | Image Description Distance | {round(image_description_vector_distance, 6)} |
                        | Tags Distance          | {round(tags_vector_distance, 6)} |
                    """)
                        # Description Vector Distance: {round(description_vector_distance, 4)}
                        # Image Description Vector Distance: {round(image_description_vector_distance, 4)}
                        # Tags Vector Distance: {round(tags_vector_distance, 4)}

            st.divider()
