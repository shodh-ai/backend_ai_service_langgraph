import sys
import json
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'curriculum_processing' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


async def get_dependencies_for_category(category_los: dict, category_name: str) -> dict | None:
    """Uses Gemini to determine prerequisites ONLY for the LOs within this category."""
    logger.info(f"Asking Gemini to analyze dependencies for category: '{category_name}'")
    
    lo_list_for_prompt = "\n".join([f"- {id}: {desc}" for id, desc in category_los.items()])

    prompt = f"""
    You are an expert curriculum designer for TOEFL preparation. 
    I have a list of Learning Objectives (LOs) that all belong to the category '{category_name}'.
    Your task is to analyze ONLY THIS list and determine the prerequisite dependencies BETWEEN THEM.

    Here is the list of LOs for this category:
    ---
    {lo_list_for_prompt}
    ---

    INSTRUCTIONS:
    For each LO in the list, determine which *other* LOs *from this same list* are necessary prerequisites.
    An LO can have zero, one, or multiple prerequisites. Be logical. For example, within 'Essay Structure', 'Writing a Topic Sentence' is a prerequisite for 'Developing a Body Paragraph'.

    Return the result as a single, valid JSON object where the top-level keys are the LO IDs. Each LO object should contain:
    - "title": The human-readable description of the LO.
    - "prerequisites": A list of strings, where each string is the `lo_id` of a prerequisite LO *from within this list*. The list can be empty.
    
    Example Output:
    {{
        "W_Ind_Thesis": {{
            "title": "Crafting a Strong Thesis Statement for TOEFL Independent Essay",
            "prerequisites": []
        }},
        "W_Ind_BodyPara_PEE": {{
            "title": "Developing a Body Paragraph (Point-Evidence-Explanation) for TOEFL Independent Essay",
            "prerequisites": ["W_Ind_Thesis"]
        }}
    }}

    Return ONLY the valid JSON object.
    """

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = await model.generate_content_async(prompt)
        cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        logger.error(f"Error processing dependencies for category '{category_name}': {e}", exc_info=True)
        return None

async def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python 2_process_category_dependencies.py <path_to_category_json_file>")
        return

    category_file_path = Path(sys.argv[1])
    if not category_file_path.exists():
        logger.error(f"File not found: {category_file_path}")
        return

    category_name = category_file_path.stem.replace("category_", "").replace("_", " ").title()
    
    logger.info(f"--- Processing dependencies for category '{category_name}' ---")
    with open(category_file_path, 'r', encoding='utf-8') as f:
        los_in_category = json.load(f)

    processed_data = await get_dependencies_for_category(los_in_category, category_name)

    if processed_data:
        # Save the result to a new file in the 'processed' subfolder
        output_filename = f"processed_{category_file_path.name}"
        output_filepath = PROCESSED_DIR / output_filename
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4)
        logger.info(f"Successfully processed and saved dependency data to: {output_filepath}")
    else:
        logger.error(f"Failed to process dependencies for {category_name}.")

if __name__ == '__main__':
    asyncio.run(main())