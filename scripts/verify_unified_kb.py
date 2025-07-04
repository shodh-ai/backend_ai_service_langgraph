import json
import logging
from pathlib import Path
import random
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_KB_PATH = PROJECT_ROOT / 'data' / 'unified' / 'unified_knowledge_base.jsonl'
NUM_SAMPLES_TO_PRINT = 20 # How many random records to print for spot-checking

def verify_unified_kb():
    """
    Parses and validates the unified knowledge base JSONL file for structure,
    content, and metadata consistency.
    """
    logger.info(f"--- Starting verification of {UNIFIED_KB_PATH} ---")

    if not UNIFIED_KB_PATH.exists():
        logger.error("Unified knowledge base file not found! Please run unify_knowledge_base.py first.")
        return

    # --- Verification Counters ---
    total_records = 0
    json_parsing_errors = 0
    missing_key_errors = 0
    empty_embedding_doc_errors = 0
    records_by_category = Counter()
    all_records = []

    # --- Pass 1: Parse and Validate Structure ---
    logger.info("--- Pass 1: Parsing JSON and validating record structure ---")
    with open(UNIFIED_KB_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            total_records += 1
            # Check for JSON validity
            try:
                record = json.loads(line)
                all_records.append(record) # Store for spot-checking later
            except json.JSONDecodeError:
                logger.error(f"L.{i}: JSON Parsing Error. Line content (first 100 chars): {line[:100]}")
                json_parsing_errors += 1
                continue

            # Check for required top-level keys
            if "document_for_embedding" not in record or "metadata" not in record:
                logger.error(f"L.{i}: Structural Error - Record missing 'document_for_embedding' or 'metadata' key.")
                missing_key_errors += 1
                continue
                
            # Check if embedding document is empty
            if not record["document_for_embedding"] or not isinstance(record["document_for_embedding"], str):
                logger.error(f"L.{i}: Content Error - 'document_for_embedding' is empty or not a string.")
                empty_embedding_doc_errors += 1
            
            # Check metadata structure and log categories
            if isinstance(record.get("metadata"), dict):
                category = record["metadata"].get("category", "UNCATEGORIZED")
                records_by_category[category] += 1
            else:
                logger.error(f"L.{i}: Structural Error - 'metadata' field is not a dictionary.")
                missing_key_errors += 1

    # --- Pass 2: Print Summary Report ---
    logger.info("--- Pass 2: Summary Report ---")
    print("\n" + "="*50)
    print("KNOWLEDGE BASE VERIFICATION SUMMARY")
    print("="*50)
    print(f"Total Records Processed: {total_records}")
    print(f"JSON Parsing Errors: {json_parsing_errors}")
    print(f"Structural Errors (missing keys): {missing_key_errors}")
    print(f"Records with Empty Embedding Docs: {empty_embedding_doc_errors}")
    print("\n--- Records Found by Category ---")
    if records_by_category:
        for category, count in sorted(records_by_category.items()):
            print(f"- {category}: {count} records")
    else:
        print("No categories found.")
    
    # Check for critical planning categories
    if 'curriculum_planning' not in records_by_category:
        logger.warning("CRITICAL WARNING: No documents with category 'curriculum_planning' were found. The CurriculumNavigatorNode's RAG will fail.")
    if 'pedagogical_sequencing' not in records_by_category:
        logger.warning("CRITICAL WARNING: No documents with category 'pedagogical_sequencing' were found. The PedagogicalStrategyPlannerNode's RAG will fail.")
    
    # --- Pass 3: Spot-Check Random Samples ---
    if all_records:
        print("\n" + "="*50)
        print("SPOT-CHECKING RECORDS")
        print("="*50)
        
        # --- NEW: Let's find and print one specific 'teaching' record ---
        teaching_sample = next((rec for rec in all_records if rec.get("metadata", {}).get("category") == "teaching"), None)
        
        if teaching_sample:
            print("\n--- DETAILED SPOT-CHECK OF ONE 'teaching' RECORD ---")
            print(f"CATEGORY: {teaching_sample.get('metadata', {}).get('category')}")
            print(f"LESSON_ID (in metadata): {teaching_sample.get('metadata', {}).get('lesson_id')}") # <-- Print the lesson_id
            print("\n  DOCUMENT FOR EMBEDDING:")
            print(f"  ------------------------")
            print(f"  {teaching_sample.get('document_for_embedding', 'N/A')}")
            print("\n  FULL METADATA:")
            print(f"  ------------------------")
            # Print the whole metadata dictionary nicely
            print(json.dumps(teaching_sample.get('metadata', {}), indent=4))
            print("-" * 25)
    
    # Final conclusion
    print("\n" + "="*50)
    if json_parsing_errors > 0 or missing_key_errors > 0 or empty_embedding_doc_errors > 0:
        logger.error("Verification FAILED with critical errors.")
    else:
        logger.info("Verification PASSED. The file appears structurally sound.")
    print("="*50)


if __name__ == "__main__":
    verify_unified_kb()