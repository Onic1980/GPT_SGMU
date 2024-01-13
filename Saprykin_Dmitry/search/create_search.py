"""
Initializes an Azure Cognitive Search index with our custom data, using vector search 
and semantic ranking.

To run this code, you must already have a "Cognitive Search" and an "OpenAI"
resource created in Azure.
"""
import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswAlgorithmConfiguration,
    SemanticPrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticSearch,
    SemanticConfiguration,
    SemanticField, 
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile
)
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
import math
import tiktoken
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
os.environ['AZURE_OPENAI_ENDPOINT'] = AZURE_OPENAI_API_BASE
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

DATA_DIR = "./data/knowledge"


def read_header(file_path: str) -> str:
    with open(file_path, "r") as f:
        lines = f.readlines()
    return "\n".join(lines[:3]) + "\n"


def load_and_split_documents() -> list[dict]:
    """
    Loads our documents from disc and split them into chunks.
    Returns a list of dictionaries.
    """
    loader = DirectoryLoader(
        DATA_DIR, loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    docs = loader.load()
    print(f"loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    split_docs = splitter.split_documents(docs)
    print(f"split into {len(split_docs)} documents")

    final_docs = []
    for i, doc in enumerate(split_docs):
        header = read_header(doc.metadata["source"])
        doc_dict = {
            "id": str(i),
            "content": header + doc.page_content,
            "sourcefile": os.path.basename(doc.metadata["source"]),
        }
        final_docs.append(doc_dict)

    return final_docs


def get_index(name: str) -> SearchIndex:
    """
    Returns an Azure Cognitive Search index with the given name.
    """
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="sourcefile", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, 
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    semantic_config = SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=None,
                    keywords_fields=[],
                    content_fields=[SemanticField(field_name="content")]
                ),
            )

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            )
        ]
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])
    
    index = SearchIndex(
        name=name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )

    return index


def initialize(search_index_client: SearchIndexClient):
    """
    Initializes an Azure Cognitive Search index with our custom data, using vector
    search.
    """
    docs = load_and_split_documents()

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_sizes = [len(encoding.encode(doc["content"])) for doc in docs]
    batch_size = 16
    num_batches = math.ceil(len(docs) / batch_size)

    print(f"embedding {len(docs)} documents in {num_batches} batches of {batch_size}. using embedding deployment {AZURE_OPENAI_EMBEDDING_DEPLOYMENT}")
    print(f"Total tokens: {sum(token_sizes)}, average tokens: {int(sum(token_sizes) / len(token_sizes))}")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(docs))
        batch_docs = docs[start_idx:end_idx]
        response = openai.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_MODEL,
            input=[doc["content"] for doc in batch_docs]
        )
        
        embeddings = response.data

        for j, doc in enumerate(batch_docs):
            doc["embedding"] = embeddings[j].embedding
            
        print(f"batch {i} embedding succes")

    print(f"creating index {AZURE_SEARCH_INDEX_NAME}")
    index = get_index(AZURE_SEARCH_INDEX_NAME)
    search_index_client.create_or_update_index(index)

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    print(f"uploading {len(docs)} documents to index {AZURE_SEARCH_INDEX_NAME}")
    search_client.upload_documents(docs)


def delete(search_index_client: SearchIndexClient):
    """
    Deletes the Azure Cognitive Search index.
    """
    print(f"deleting index {AZURE_SEARCH_INDEX_NAME}")
    search_index_client.delete_index(AZURE_SEARCH_INDEX_NAME)


def main():
    load_dotenv()

    openai.api_type = AZURE_OPENAI_API_TYPE
    openai.api_base = AZURE_OPENAI_API_BASE
    openai.api_version = AZURE_OPENAI_API_VERSION
    openai.api_key = AZURE_OPENAI_API_KEY

    search_index_client = SearchIndexClient(
        AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    delete(search_index_client)
    initialize(search_index_client)


if __name__ == "__main__":
    main()
