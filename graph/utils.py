# FINAL, ROBUST graph/utils.py

import logging
import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# --- Configuration ---
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "tutor_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3

# --- Global variables to hold the client and collection ---
# We still want to reuse the connection if it's already established.
_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None

def get_chroma_collection() -> Optional[chromadb.Collection]:
    """
    Initializes and returns the ChromaDB collection.
    Uses a singleton pattern to avoid reconnecting on every call.
    This is more resilient than initializing at the top level.
    """
    global _client, _collection

    # If the connection already exists, just return it.
    if _collection is not None:
        return _collection

    # If it doesn't exist, try to create it.
    try:
        logger.info("Attempting to initialize ChromaDB connection...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, DB_DIRECTORY)
        
        if not os.path.exists(db_path):
            logger.error(f"ChromaDB directory not found at: {db_path}. Please run the ingestion script.")
            return None

        _client = chromadb.PersistentClient(path=db_path)
        
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=sentence_transformer_ef
        )
        logger.info(f"ChromaDB connection successful. Collection '{COLLECTION_NAME}' loaded.")
        return _collection

    except Exception as e:
        logger.error(f"Failed to initialize or get ChromaDB collection: {e}", exc_info=True)
        # Ensure they are reset to None on failure
        _client = None
        _collection = None
        return None

async def query_knowledge_base(query_string: str, category: str, lesson_id: str = None) -> List[Dict[str, Any]]:
    """
    Queries the ChromaDB knowledge base with enhanced filtering.

    Args:
        query_string (str): The user's query.
        category (str): The primary category to filter by (e.g., 'teaching').
        lesson_id (str, optional): The specific lesson_id to filter by. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of retrieved documents.
    """
    log_message = f"Querying KB for category '{category}'" 
    if lesson_id:
        log_message += f" and lesson_id '{lesson_id}'"
    log_message += f" with query: '{query_string[:50]}...'"
    logger.info(log_message)

    try:
        collection = get_chroma_collection()
        
        if collection is None:
            logger.error("Cannot perform RAG because ChromaDB collection is not available.")
            return []

        # --- Dynamic Filter Construction ---
        filter_conditions = {"category": category}
        if lesson_id:
            # Using $and to require both conditions to be met
            filter_conditions = {
                "$and": [
                    {"category": {"$eq": category}},
                    {"lesson_id": {"$eq": lesson_id}}
                ]
            }
        
        results = collection.query(
            query_texts=[query_string],
            n_results=TOP_K_RESULTS,
            where=filter_conditions
        )

        retrieved_documents = results.get('metadatas', [[]])[0]
        
        logger.info(f"Query successful. Retrieved {len(retrieved_documents)} documents.")
        return retrieved_documents

    except Exception as e:
        logger.error(f"An error occurred during knowledge base query: {e}", exc_info=True)
        return []