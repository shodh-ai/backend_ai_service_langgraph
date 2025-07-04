# scripts/unify_knowledge_base.py

import pandas as pd
import os
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data' / 'source_cta'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'unified'
OUTPUT_PATH = OUTPUT_DIR / 'unified_knowledge_base.jsonl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_FILES = {
    "pedagogical_data.csv": "pedagogy",
    "cowriting_data.csv": "cowriting",
    "modelling_data.csv": "modelling",
    "scaffolding_data.csv": "scaffolding",
    "teaching_data.csv": "teaching",
    "feedback_data.csv": "feedback",
}

def create_embedding_document(row: pd.Series, doc_type: str) -> str:
    """
    Intelligently creates a single text document for embedding based on the DOC_TYPE.
    This function is now more explicit about what kind of document it's creating.
    """
    text_parts = []
    try:
        if doc_type == "pedagogy_macro":
            # This document is for the HIGH-LEVEL Curriculum Navigator (Macro Planner).
            # It focuses on the student's profile and the REASONING for the plan.
            text_parts.append("This document describes a high-level pedagogical plan for a specific student profile.")
            text_parts.append(f"Student Profile Summary:")
            text_parts.append(f"  Goal: {row.get('Answer One', '')}")
            text_parts.append(f"  Confidence/Struggle: {row.get('Answer Two', '')}")
            text_parts.append(f"  Attitude: {row.get('Answer Three', '')}")
            text_parts.append(f"  Impression: {row.get('Initial Impression', '')}")
            text_parts.append(f"  Comfort Level: {row.get('Estimated Overall English Comfort Level', '')}")
            try:
                pedagogy_plan = json.loads(row.get('Pedagogy', '{}'))
                text_parts.append(f"Expert Teacher's Reasoning for Overall Plan: {pedagogy_plan.get('reasoning', '')}")
            except (json.JSONDecodeError, TypeError): pass

        elif doc_type == "pedagogy_meso":
            # This document is for the LESSON-LEVEL Pedagogical Strategy Planner (Meso Planner).
            # It focuses on the *sequence* of teaching modalities.
            text_parts.append("This document describes a sequence of learning modalities for a specific student profile and learning topic.")
            text_parts.append(f"For a student whose profile is: Initial Impression '{row.get('Initial Impression', '')}', Comfort Level '{row.get('Estimated Overall English Comfort Level', '')}'.")
            try:
                pedagogy_plan = json.loads(row.get('Pedagogy', '{}'))
                steps = pedagogy_plan.get('steps', [])
                first_teaching_step = next((step for step in steps if step.get('type') == 'Teaching'), None)
                lo_topic = first_teaching_step.get('topic') if first_teaching_step else "General Skills"

                text_parts.append(f"For the Learning Objective Topic of: {lo_topic}.")
                text_parts.append(f"The recommended sequence of teaching modalities is: {' -> '.join([f'{s.get("type", "N/A")}({s.get("task", "N/A")})' for s in steps])}")
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse 'Pedagogy' JSON for meso doc on row {row.name}. Content: '{row.get('Pedagogy', '')}'")
                pass # Still create a partial doc with student profile.

        elif doc_type == "teaching":
            text_parts.append(f"Teaching strategy for Learning Objective: {row.get('LEARNING_OBJECTIVE', '')}")
            text_parts.append(f"Explanation for a {row.get('STUDENT_PROFICIENCY', '')} student: {row.get('CORE_EXPLANATION_STRATEGY', '')}")
            text_parts.append(f"Key Examples to use: {row.get('KEY_EXAMPLES', '')}")

        elif doc_type == "modelling":
            text_parts.append(f"Modeling demonstration for skill: {row.get('Learning_Objective_ID', '')}")
            text_parts.append(f"Persona: {row.get('Persona', '')}")
            text_parts.append(f"Example TOEFL Prompt for Model: {row.get('Example_Prompt_Text', '')}")
            text_parts.append(f"Think-Aloud Summary: {row.get('post_modeling_summary_and_key_takeaways', '')}")

        elif doc_type == "scaffolding":
            text_parts.append(f"Scaffolding technique for task: {row.get('Learning_Objective_Task', '')}")
            text_parts.append(f"Persona: {row.get('Persona', '')}")
            text_parts.append(f"Specific student struggle addressed: {row.get('Specific_Struggle_Point', '')}")
            text_parts.append(f"Type of scaffold chosen: {row.get('scaffold_type_selected', '')}")
            text_parts.append(f"Reasoning for this scaffold: {row.get('reasoning_for_scaffold_choice', '')}")

        elif doc_type == "cowriting":
            text_parts.append(f"Live co-writing assistance strategy for topic: {row.get('Learning_Objective_Focus', '')}")
            text_parts.append(f"Persona: {row.get('Persona', '')}")
            text_parts.append(f"Student wrote: '{row.get('Student_Written_Input_Chunk', '')}'")
            text_parts.append(f"AI Intervention Type: {row.get('Intervention_Type', '')}")
            text_parts.append(f"AI Suggestion: {row.get('AI_Spoken_or_Suggested_Text', '')}")
        
        elif doc_type == "feedback":
             text_parts.append(f"Feedback strategy for correcting error: {row.get('Error', '')}")
             text_parts.append(f"Persona: {row.get('Persona', '')}")
             text_parts.append(f"Student context: Proficiency {row.get('Proficiency', '')}, feeling {row.get('Behavior Factor', '')}")
             text_parts.append(f"Diagnosis of the problem: {row.get('Diagnose', '')}")
             text_parts.append(f"Method of explanation: {row.get('Explain Strategy', '')}")

        else: 
            logging.warning(f"No specific embedding document creation logic for doc_type: '{doc_type}'.")
            return ""

        return " \n ".join(filter(None, text_parts)).strip()
    except Exception as e:
        logging.error(f"Error creating document for doc_type '{doc_type}' on row {row.name}: {e}")
        return ""

def unify_and_standardize_cta_data():
    final_standardized_records = []
    logging.info("Starting unification and standardization process...")

    # --- Configuration ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SOURCE_DIR = PROJECT_ROOT / 'data' / 'source_cta'
    OUTPUT_DIR = PROJECT_ROOT / 'data' / 'unified'
    OUTPUT_PATH = OUTPUT_DIR / 'unified_knowledge_base.jsonl'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SOURCE_FILES = {
        "pedagogical_data.csv": "pedagogy",
        "cowriting_data.csv": "cowriting",
        "modelling_data.csv": "modelling",
        "scaffolding_data.csv": "scaffolding",
        "teaching_data.csv": "teaching",
        "feedback_data.csv": "feedback",
    }
    
    for filename, category_from_filename in SOURCE_FILES.items():
        file_path = os.path.join(SOURCE_DIR, filename)
        if not os.path.exists(file_path):
            logging.warning(f"File not found, skipping: {file_path}")
            continue
        try:
            logging.info(f"Processing '{filename}' for category '{category_from_filename}'...")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            for _, row in df.iterrows():
                metadata = {k.strip(): str(v) for k, v in row.to_dict().items() if pd.notna(v)}

                if category_from_filename == 'pedagogy':
                    # --- DUAL RECORD CREATION LOGIC ---
                    
                    # 1. Macro Planner Record
                    macro_doc = create_embedding_document(row, "pedagogy_macro")
                    if macro_doc:
                        macro_metadata = metadata.copy()
                        macro_metadata['category'] = 'curriculum_planning'
                        final_standardized_records.append({"document_for_embedding": macro_doc, "metadata": macro_metadata})

                    # 2. Meso Planner Record
                    meso_doc = create_embedding_document(row, "pedagogy_meso")
                    if meso_doc:
                        meso_metadata = metadata.copy()
                        meso_metadata['category'] = 'pedagogical_sequencing'
                        final_standardized_records.append({"document_for_embedding": meso_doc, "metadata": meso_metadata})
                
                else:
                    # --- SINGLE RECORD CREATION FOR ALL OTHER FILES ---
                    embedding_doc = create_embedding_document(row, category_from_filename)
                    if not embedding_doc:
                        logging.warning(f"Skipping row {row.name} in {filename} due to empty document.")
                        continue
                    
                    # Set the base category from the filename
                    metadata['category'] = category_from_filename
                    
                    # ====================================================================
                    # ▼▼▼ THIS IS THE CRITICAL FIX FOR THE 'teaching' CATEGORY ▼▼▼
                    # ====================================================================
                    if category_from_filename == 'teaching':
                        # This logic now ADDS the lesson_id but KEEPS the base 'teaching' category.
                        # It is NOT an overwrite.
                        learning_objective_full = str(row.get('LEARNING_OBJECTIVE', '')).strip()
                        if learning_objective_full:
                            lesson_id = learning_objective_full.split(':')[0].strip()
                            if lesson_id:
                                # We add 'lesson_id' as an *additional* piece of metadata for filtering.
                                metadata['lesson_id'] = lesson_id
                    # ====================================================================
                    # ▲▲▲ END OF CRITICAL FIX ▲▲▲
                    # ====================================================================

                    final_standardized_records.append({
                        "document_for_embedding": embedding_doc,
                        "metadata": metadata
                    })

        except Exception as e:
            logging.error(f"Failed to process file {filename}. Error: {e}", exc_info=True)

    logging.info(f"Saving {len(final_standardized_records)} records to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for record in final_standardized_records:
            f.write(json.dumps(record) + '\n')
    logging.info("Unified knowledge base saved successfully.")


if __name__ == "__main__":
    unify_and_standardize_cta_data()