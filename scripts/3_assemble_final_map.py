import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'curriculum_processing' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'backend_ai_service_langgraph' / 'knowledge_base_content'
FINAL_MAP_FILE = OUTPUT_DIR / 'curriculum_map.json'

def assemble_map():
    if not PROCESSED_DIR.exists():
        logger.error(f"Processed directory not found: {PROCESSED_DIR}")
        return
    
    final_curriculum_map = {}
    processed_files = list(PROCESSED_DIR.glob("processed_*.json"))
    
    if not processed_files:
        logger.error("No processed category files found to assemble.")
        return

    logger.info(f"Found {len(processed_files)} processed category files. Assembling final map...")

    for file_path in processed_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                category_data = json.load(f)
                # We need to add the 'category' key back into the final object
                category_name = file_path.stem.replace("processed_category_", "").replace("_", " ").title()
                for lo_id, lo_data in category_data.items():
                    if lo_id not in final_curriculum_map:
                        lo_data['category'] = category_name # Add the category field
                        final_curriculum_map[lo_id] = lo_data
                    else:
                        logger.warning(f"Duplicate LO_ID '{lo_id}' found. Keeping first instance.")
        except Exception as e:
            logger.error(f"Failed to read or process file '{file_path}': {e}", exc_info=True)

    if not final_curriculum_map:
        logger.error("Assembly failed. Final map is empty.")
        return

    # Save the final, unified map
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_curriculum_map, f, indent=4, sort_keys=True)
        
    logger.info(f"SUCCESS! Assembled final curriculum map with {len(final_curriculum_map)} LOs.")
    logger.warning("Please proceed to the FINAL HUMAN REVIEW of curriculum_map.json.")

if __name__ == "__main__":
    assemble_map()