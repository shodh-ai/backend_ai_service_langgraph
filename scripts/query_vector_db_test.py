import chromadb
from chromadb.utils import embedding_functions
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tutor_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TEST_QUERIES = [
{
"description": "Macro Planning: What to do for a new, nervous student?",
"query_text": "A new beginner student feels nervous but is motivated to improve their speaking and writing for grad school.",
"filter": {"category": "curriculum_planning"}
},
{
"description": "Meso Planning: How to sequence a lesson for 'Thesis Statements'?",
"query_text": "What is the best sequence of teaching modalities for explaining Thesis Statements to a beginner?",
"filter": {"category": "pedagogical_sequencing"}
},
{
"description": "Teaching Content: How does 'The Structuralist' explain P-E-E paragraphs?",
"query_text": "The Structuralist persona needs to teach the Point-Evidence-Explanation paragraph structure.",
"filter": {"category": "teaching"} # You could filter by lesson_id if you know it
},
{
"description": "Modeling: How to show someone how to handle an Integrated Writing Task?",
"query_text": "Model how to synthesize reading and listening points for the integrated writing task.",
"filter": {"category": "modelling"}
},
{
"description": "Scaffolding: Student is stuck starting a speaking response.",
"query_text": "A beginner student is having trouble starting their speaking response and is showing hesitation and frustration.",
"filter": {"category": "scaffolding"}
},
{
"description": "Feedback: Student has a fluency problem.",
"query_text": "The student has a fluency error with many pauses and filler words like 'um'.",
"filter": {"category": "feedback"}
}
]
def print_query_results(results: dict):

    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i]

        print(f"\n  --- Result {i+1} (ID: {doc_id}, Distance: {distance:.4f}) ---")
        print(f"  CATEGORY: {metadata.get('category', 'N/A')}")
        print(f"  SOURCE LO: {metadata.get('Learning_Objective_ID') or metadata.get('Learning_Objective_Task') or metadata.get('Error', 'N/A')}")
        print("\n  RETRIEVED DOCUMENT FOR EMBEDDING:")
        print(f"  {document}")
# print("\n  FULL METADATA (SAMPLE):")
# print(f"  {json.dumps({k: v for k, v in metadata.items() if k != 'document_for_embedding'}, indent=4)}")
print("-" * 20)

def test_vector_db_queries():
    logger.info(f"--- Starting Vector DB Query Test ---")
    logger.info(f"Connecting to ChromaDB at: {DB_DIRECTORY}")
    if not DB_DIRECTORY.exists():
        logger.error(f"ChromaDB directory not found! Please run ingest.py first.")
        return

# --- 1. Initialize Client and Get Collection ---
try:
    client = chromadb.PersistentClient(path=str(DB_DIRECTORY))
    
    # This uses a pre-trained model on your machine. No API calls are made here.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(
        name=COLLECTION_NAME, 
        embedding_function=sentence_transformer_ef
    )
    collection_count = collection.count()
    logger.info(f"Successfully connected to collection '{COLLECTION_NAME}' with {collection_count} items.")
    if collection_count == 0:
        logger.error("Collection is empty! Ingestion may have failed.")
except Exception as e:
    logger.error(f"Failed to connect to ChromaDB or get collection. Error: {e}", exc_info=True)

# --- 2. Run Test Queries ---
for i, test in enumerate(TEST_QUERIES):
    print("\n" + "="*50)
    print(f"EXECUTING TEST QUERY {i+1}: {test['description']}")
    print("="*50)
    
    try:
        results = collection.query(
            query_texts=[test["query_text"]],
            n_results=3, # Get top 3 results
            where=test["filter"] # Apply the category filter
        )
        
        if not results or not results['ids'][0]:
            logger.warning("Query returned NO results. This might indicate an issue with the data for this category or the query itself.")
        else:
            print_query_results(results)

    except Exception as e:
        logger.error(f"An error occurred during query for '{test['description']}': {e}", exc_info=True)

print("\n" + "="*50)
print("QUERY TEST COMPLETE.")
print("="*50)