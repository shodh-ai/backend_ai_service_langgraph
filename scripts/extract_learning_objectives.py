import pandas as pd
import os
import json
from pathlib import Path
import logging

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use Pathlib for OS-agnostic path handling
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA_DIR = PROJECT_ROOT / 'data' / 'source_cta'
OUTPUT_DIR = PROJECT_ROOT / 'knowledge_base_content'
LO_LIST_OUTPUT_FILE = OUTPUT_DIR / 'master_learning_objectives.json'

# This dictionary is the "source of truth" for mapping your CSVs to the data we need.
# It defines which file to read, which column contains the LO's unique ID,
# and which column contains a human-readable description for it.
# Files not listed here will be ignored.
CTA_SOURCE_CONFIG = {
    "teaching_data.csv": {
        "id_col": "LEARNING_OBJECTIVE",
        "desc_col": "LESSON_FOR_STUDENT"
    },
    "modelling_data.csv": {
        # IMPORTANT: This assumes you have MANUALLY ADDED this column to your CSV file!
        "id_col": "Learning_Objective_ID",
        "desc_col": "pre_modeling_setup_script"
    },
    "scaffolding_data.csv": {
        "id_col": "Learning_Objective_Task",
        "desc_col": "scaffold_type_selected" # Using the type of scaffold as description
    },
    "cowriting_data.csv": {
        "id_col": "Learning_Objective_Focus",
        "desc_col": "Writing_Task_Context_Section"
    },
    "feedback_data.csv": {
        # Assuming your feedback data uses the error type as the "thing to learn"
        "id_col": "Error",
        "desc_col": "Diagnose"
    }
}

def extract_unique_los() -> dict:
    """
    Extracts a unique set of Learning Objectives (LOs) from all configured CTA data sources.
    It reads each specified CSV, finds the configured columns, and builds a de-duplicated
    master list of all skills, concepts, and topics your AI tutor knows about.

    Returns:
        A dictionary mapping unique LO IDs to their descriptions.
    """
    all_los = {}  # Using a dict {lo_id: lo_description} de-duplicates by ID automatically
    logger.info("Starting extraction of unique Learning Objectives from all configured CTA sources...")

    for filename, config in CTA_SOURCE_CONFIG.items():
        file_path = SOURCE_DATA_DIR / filename
        
        if not file_path.exists():
            logger.warning(f"File '{filename}' not found at '{file_path}', skipping.")
            continue

        try:
            logger.info(f"Processing '{filename}'...")
            df = pd.read_csv(file_path)
            # Sanitize column names to prevent issues with leading/trailing spaces
            df.columns = df.columns.str.strip()

            id_col = config["id_col"]
            desc_col = config["desc_col"]

            if id_col not in df.columns:
                logger.error(f"FATAL: Configured ID column '{id_col}' not found in '{filename}'. This file cannot be processed for LOs. Please check the file or configuration. Skipping file.")
                continue
            
            if desc_col not in df.columns:
                logger.warning(f"Description column '{desc_col}' not found in '{filename}'. Will use ID as description.")
                desc_col = id_col  # Fallback to using the ID as its own description

            # Iterate over each row in the dataframe
            for _, row in df.iterrows():
                lo_id = row.get(id_col)
                lo_desc = row.get(desc_col)

                # Ensure the data is a valid string and not empty/NaN
                if pd.notna(lo_id) and isinstance(lo_id, str) and lo_id.strip():
                    cleaned_id = lo_id.split(':')[0].strip() # Takes the part before a ':', e.g., "SPK_GEN_FLU_001"
                    
                    if cleaned_id not in all_los:
                        # Use description if it's valid, otherwise fallback to the ID itself
                        description = lo_desc if pd.notna(lo_desc) and isinstance(lo_desc, str) else cleaned_id
                        all_los[cleaned_id] = description.strip()

        except Exception as e:
            logger.error(f"An error occurred while processing '{filename}': {e}", exc_info=True)
            continue # Move to the next file

    logger.info(f"Extraction complete. Found {len(all_los)} unique Learning Objectives.")
    
    if not all_los:
        logger.error("No learning objectives were extracted. Please check your CSV files and `CTA_SOURCE_CONFIG` in this script.")
        return {}
        
    # Save the final, master list of LOs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LO_LIST_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_los, f, indent=4)
    logger.info(f"Master list of Learning Objectives saved to: {LO_LIST_OUTPUT_FILE}")
    
    return all_los

if __name__ == "__main__":
    # Load environment variables if needed by any downstream imports (good practice)
    # load_dotenv() 
    extract_unique_los()