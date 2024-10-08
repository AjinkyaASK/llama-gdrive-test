import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema
from llama_index.core import VectorStoreIndex
from llama_index.readers.google import GoogleDriveReader


os.environ["OPENAI_API_KEY"] = ""

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "gdrive", "prefix": "doc"},
        # we can customize the fields below that are needed for indexing
        "fields": [
            # mandatory fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # required vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)

# Make sure the redis server is running on port 6379
vector_store = RedisVectorStore(
    schema=custom_schema,
    redis_url="redis://localhost:6379",
)

if vector_store.index_exists():
    vector_store.delete_index()
    
cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

loader = GoogleDriveReader()

def load_data(folder_id: str):
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        doc.id_ = doc.metadata["file_name"]
    return docs


docs = load_data(folder_id="MY_GDRIVE_FOLDER_ID")

nodes = pipeline.run(documents=docs)
print(f"Ingested {len(nodes)} Nodes")