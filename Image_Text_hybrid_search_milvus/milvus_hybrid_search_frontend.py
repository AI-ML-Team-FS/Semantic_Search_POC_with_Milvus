import streamlit as st
import pandas as pd
from PIL import Image
import os
import subprocess
import os
import asyncio
from huggingface_hub import snapshot_download
from pymilvus import MilvusClient, DataType
import subprocess
from pymilvus import Function, FunctionType
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import milvus_methods_version_3
from tqdm import tqdm

def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", 
                                #   low_cpu_mem_usage=True,
                                  torch_dtype=torch.float32
                                  ).to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

os.environ["STREAMLIT_WATCH_DIRS"] = ""

# collection_name = "SemanticSearch"
collection_name = "SemanticSearch_bgremoved"

# subprocess.run(["docker", "compose", "up", "-d"], check=True)

df = pd.read_csv('DATA/products_with_tags_and_description_seperated.csv')

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

if not client.has_collection(collection_name=collection_name):
    # client.drop_collection(collection_name=collection_name)
    
    st.write(f"Creating collection {collection_name}")

    schema = MilvusClient.create_schema(
        auto_id=False,
    )

    schema.add_field(field_name="Handle", datatype=DataType.VARCHAR, max_length=512, is_primary=True)
    schema.add_field(field_name="Title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="Type", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='craft', datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="fabric", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="Product_Types", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="Tags", datatype=DataType.VARCHAR, max_length=256, enable_analyzer=True)
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=1024, enable_analyzer=True)
    schema.add_field(field_name="Description", datatype=DataType.VARCHAR, max_length=1024, enable_analyzer=True)

    schema.add_field(field_name="description_vector", datatype=DataType.FLOAT_VECTOR, dim = 768)
    schema.add_field(field_name="image_description_vector", datatype=DataType.FLOAT_VECTOR, dim = 768)
    schema.add_field(field_name="tags_vector", datatype=DataType.FLOAT_VECTOR, dim = 768)
    schema.add_field(field_name="image_dense", datatype=DataType.FLOAT_VECTOR, dim = 768)
    schema.add_field(field_name="description_vector_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="image_description_vector_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="tags_vector_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # Add function to schema
    bm25_function_1 = Function(
        name="description_bm25_emb",
        input_field_names=["description"],
        output_field_names=["description_vector_sparse"],
        function_type=FunctionType.BM25,
    )
    bm25_function_2 = Function(
        name="image_description_bm25_emb",
        input_field_names=["Description"],
        output_field_names=["image_description_vector_sparse"],
        function_type=FunctionType.BM25,
    )
    bm25_function_3 = Function(
        name="tags_bm25_emb",
        input_field_names=["Tags"],
        output_field_names=["tags_vector_sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function_1)
    schema.add_function(bm25_function_2)
    schema.add_function(bm25_function_3)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="description_vector",
        index_type="FLAT", 
        index_name="description_vector_index",
        metric_type="COSINE", 
    )
    index_params.add_index(
        field_name="image_description_vector", 
        index_type="FLAT",
        index_name="image_description_vector_index",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="tags_vector", # Name of the vector field to be indexed
        index_type="FLAT", # Type of the index to create
        index_name="tags_vector_index", # Name of the index to create
        metric_type="COSINE", # Metric type used to measure similarity
    )
    index_params.add_index(
        field_name="image_dense",
        index_name="image_dense_index",
        index_type="FLAT",
        metric_type="COSINE"
    )

    index_params.add_index(
        field_name="image_description_vector_sparse",
        index_name="image_description_vector_sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
    )
    index_params.add_index(
        field_name="description_vector_sparse",
        index_name="description_vector_sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
    )
    index_params.add_index(
        field_name="tags_vector_sparse",
        index_name="tags_vector_sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}, # or "DAAT_WAND" or "TAAT_NAIVE"
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    res = client.get_load_state(
    collection_name=collection_name
    )

    df['description'] = df['description'].fillna('')
    df['Description'] = df['Description'].fillna('')
    df['color'] = df['color'].fillna('')
    df['craft'] = df['craft'].fillna('')
    df['fabric'] = df['fabric'].fillna('')
    df['Product Types'] = df['Product Types'].fillna('')

    df['Tags'] = df.apply(milvus_methods_version_3.to_string, axis=1)

    df['description_vector'] = df['description'].apply(lambda x: milvus_methods_version_3.get_text_vector(x, model, processor))
    df['image_description_vector'] = df['Description'].apply(lambda x: milvus_methods_version_3.get_text_vector(x, model, processor))
    df['Tags_vector'] = df['Tags'].apply(lambda x: milvus_methods_version_3.get_text_vector(x, model, processor))

    image_vectors = []

    for handle in tqdm(df['Handle']):
        image_path = f"DATA/jpeg_ROI_images/{handle}.jpg"
        try:
            vec = milvus_methods_version_3.get_image_vector(image_path, model, processor)
        except Exception as e:
            print(f"Failed to process {handle}: {e}")
            vec = [0.0] * 768  # Or skip, or use None
        image_vectors.append(vec)
    df['image_dense'] = image_vectors

    df_copy = df.copy()
    df_copy.drop(columns=['Unnamed: 0'], inplace = True)
    df_copy.rename(columns={'Tags_vector': 'tags_vector'}, inplace=True)
    df_copy.rename(columns={'Product Types': 'Product_Types'}, inplace=True)
    df_copy['craft'] = df_copy['craft'].fillna("Unknown").astype(str)

    data = df_copy.to_dict(orient='records')
    batch_size = 10 # You can adjust this based on performance/memory

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        res = client.insert(
            collection_name=collection_name,
            data=batch
        )
        st.write(i)

st.title("Semantic Search Milvus v3 with Hybrid Search and Image: Sue Mue POC")

text_flag = False
if st.checkbox("Text search", value=False):
    user_query = st.text_input("Enter your query: ")
    text_flag = True

image_flag = False
if st.checkbox("Search by Image: ", value=False):
    user_image = st.file_uploader("Upload an image: ", type=["jpg", "png", "jpeg"])
    image_flag = True

if st.button("Search"):
    with st.spinner("Searching for products..."):
        st.write("Searching for products...")
        
        TOP_K = 5

        if image_flag and text_flag:
            products = milvus_methods_version_3.hybrid_search_image_and_text(
                user_query=user_query,
                user_image=user_image,
                client=client,
                model=model,
                processor=processor,
                top_k=int(TOP_K)
                )
            st.write("Search prodcuts with text and image:", products)

        elif text_flag and not image_flag:
            products = milvus_methods_version_3.hybrd_search_text_only(user_query=user_query, model=model, client=client,processor=processor,top_k=int(TOP_K))
            st.write("Search prodcuts with Text only:", products)

        elif image_flag and not text_flag:
            products = milvus_methods_version_3.hybrid_search_image_only(
                user_image=user_image,
                client=client,
                model=model,
                processor=processor,
                top_k=int(TOP_K)
                )
            st.write("Search prodcuts with image only:", products)

        # handles = [entry.get("Handle") or entry.get("handle") for entry in products if entry.get("Handle") or entry.get("handle")]
        handles = [entry.get("handle") for entry in products if entry.get("handle")]

        products = milvus_methods_version_3.search_products(handles, df)
        # st.write("Products using method search_products:", products)

        results_for_csv = []
        for i, product in enumerate(products):

            data = product["data"]
            image = product["image"]

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

                    """)

            for i, product in enumerate(products):
                data = product["data"]
                
                results_for_csv.append({
                    "Query": user_query if text_flag else None,
                    "Handle": data["Handle"],
                    "Title": data["Title"],
                    "Product Type": data["Product Types"],
                    "Color": data["color"],
                    "Fabric": data["fabric"],
                    "Craft": data["craft"],
                    "Official Description": data["description"],
                    "LLM Image Description": data["Description"]
                })

            st.divider()
    if text_flag:
        csv_output_path = f"Search_Results/{user_query.replace(' ', '_')}.csv"
        pd.DataFrame(results_for_csv).to_csv(csv_output_path, index=False)
