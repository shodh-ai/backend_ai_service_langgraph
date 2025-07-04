import pandas as pd
import os
import json
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'source_cta'
OUTPUT_DIR = PROJECT_ROOT / 'curriculum_processing' # New intermediate directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (Re-use your CTA_SOURCE_CONFIG and extract_unique_los function from previous script)
CTA_SOURCE_CONFIG = { # Make sure this is accurate for your files
    "teaching_data.csv":      {"id_col": "LEARNING_OBJECTIVE", "desc_col": "LESSON_FOR_STUDENT"},
    "modelling_data.csv":         {"id_col": "Learning_Objective_ID", "desc_col": "pre_modeling_setup_script"},
    "scaffolding_data.csv":       {"id_col": "Learning_Objective_Task", "desc_col": "scaffold_type_selected"},
    "cowriting_data.csv":         {"id_col": "Learning_Objective_Focus", "desc_col": "Writing_Task_Context_Section"},
    "feedback_data.csv":          {"id_col": "Error", "desc_col": "Diagnose"}
}

def extract_unique_los() -> dict:
    # ... (Paste the full extract_unique_los function from the previous answer here) ...
    all_los = {}
    logger.info("Starting extraction of unique Learning Objectives...")
    for filename, config in CTA_SOURCE_CONFIG.items():
        path = DATA_DIR / filename
        if not path.exists(): continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        id_col, desc_col = config.get("id_col"), config.get("desc_col")
        if not id_col or id_col not in df.columns: continue
        if desc_col not in df.columns: desc_col = id_col
        for _, row in df.iterrows():
            lo_id = str(row[id_col]).strip() if pd.notna(row.get(id_col)) else None
            if lo_id and lo_id.lower() != 'nan':
                cleaned_id = lo_id.split(':')[0].strip()
                if cleaned_id and cleaned_id not in all_los:
                    lo_desc = str(row.get(desc_col, cleaned_id)).strip()
                    all_los[cleaned_id] = lo_desc
    logger.info(f"Extraction complete. Found {len(all_los)} unique LOs.")
    return all_los


async def categorize_los(los_dict: dict) -> dict:
    """Uses Gemini to group a list of LOs into logical categories."""
    logger.info("Asking Gemini to categorize all learning objectives...")

    lo_list_for_prompt = "\n".join([f"- {id}: {desc}" for id, desc in los_dict.items()])
    
    prompt = f"""
    You are an expert curriculum designer for TOEFL preparation. 
    Analyze this list of Learning Objectives (LOs) and group each one into a single, logical category.
    
    Example Categories: 'Foundational Essay Structure', 'Advanced Essay Techniques', 'Speaking Fluency Skills', 'Integrated Task Skills', 'Core Grammar', 'Academic Vocabulary', 'General Test Strategies'. You can create new categories if necessary.

    Here is the list of LOs:
    ---
    {lo_list_for_prompt}
    ---

    Return your answer as a single, valid JSON object where keys are the LO IDs and the value is the assigned category string.

    Example Output for a few items:
    {{
        "W_Ind_Thesis": "Foundational Essay Structure",
        "SPK_Q1_Model_PREP": "Speaking Fluency Skills",
        "G_PastTense": "Core Grammar"
    }}
    
    Return ONLY the valid JSON object.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        response = await model.generate_content_async(prompt)
        cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        logger.error(f"Error during LLM categorization: {e}", exc_info=True)
        return None

async def main():
    # Step 1: Extract all unique LOs
    learning_objectives = extract_unique_los()
    if not learning_objectives:
        return

    # Step 2: Use an LLM to categorize them
    lo_categories = await categorize_los(learning_objectives)
    if not lo_categories:
        logger.error("Failed to categorize Learning Objectives. Aborting.")
        return

    # Step 3: Group LOs by their new category and save to separate files
    grouped_los = {}
    for lo_id, category in lo_categories.items():
        if category not in grouped_los:
            grouped_los[category] = {}
        # Ensure the description is carried over
        if lo_id in learning_objectives:
            grouped_los[category][lo_id] = learning_objectives[lo_id]

    logger.info(f"Grouped LOs into {len(grouped_los)} categories.")
    for category, los_in_category in grouped_los.items():
        # Sanitize category name for filename
        filename = f"category_{category.replace(' ', '_').replace('&', 'and').lower()}.json"
        filepath = OUTPUT_DIR / filename

        # Ensure the parent directory exists before writing the file
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(los_in_category, f, indent=4)
        logger.info(f"Saved category file: {filepath}")
    
    logger.info("\n--- CATEGORIZATION COMPLETE ---")
    logger.info("Next Step: Run '2_process_category_dependencies.py' on each generated category file.")
    logger.info("You can run this in parallel in separate terminals. Example:")
    logger.info("python scripts/2_process_category_dependencies.py curriculum_processing/category_core_grammar.json")
    logger.info("python scripts/2_process_category_dependencies.py curriculum_processing/category_speaking_fluency_skills.json")


if __name__ == '__main__':
    asyncio.run(main())