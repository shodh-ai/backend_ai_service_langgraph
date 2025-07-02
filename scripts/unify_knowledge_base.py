# scripts/unify_knowledge_base.py (Final Version)

import pandas as pd
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_embedding_document(row: pd.Series, category: str) -> str:
    """
    Intelligently creates a single text document for embedding based on the category.
    This function is the heart of the standardization process, tailored to each
    unique CSV schema.
    """
    text_parts = []
    
    try:
        if category == "cowriting":
            text_parts.append(f"Learning Objective: {row.get('Learning_Objective_Focus', '')}")
            text_parts.append(f"Student Input: {row.get('Student_Written_Input_Chunk', '')}")
            text_parts.append(f"Struggle: {row.get('Immediate_Assessment_of_Input', '')}")
            text_parts.append(f"AI Suggestion: {row.get('AI_Spoken_or_Suggested_Text', '')}")
            text_parts.append(f"AI Rationale: {row.get('Rationale_for_Intervention_Style', '')}")
            
        elif category == "modelling":
            text_parts.append(f"Task Prompt being modeled: {row.get('Example_Prompt_Text', '')}")
            text_parts.append(f"Intended for student with this struggle: {row.get('Student_Struggle_Context', '')}")
            text_parts.append(f"Setup Script: {row.get('pre_modeling_setup_script', '')}")
            text_parts.append(f"Key takeaways from the model: {row.get('post_modeling_summary_and_key_takeaways', '')}")

        elif category == "pedagogy":
            text_parts.append(f"Student Answers: {row.get('Answer One', '')} {row.get('Answer Two', '')} {row.get('Answer Three', '')}")
            text_parts.append(f"Teacher's initial impression: {row.get('Initial Impression', '')}")
            text_parts.append(f"Student's assessed strengths: {row.get('Speaking Strengths', '')}")
            # The 'Pedagogy' column contains a JSON string. We want the reasoning text inside it.
            try:
                pedagogy_plan = json.loads(row.get('Pedagogy', '{}'))
                text_parts.append(f"Expert reasoning for the curriculum plan: {pedagogy_plan.get('reasoning', '')}")
            except (json.JSONDecodeError, TypeError):
                text_parts.append(f"Pedagogical Plan: {row.get('Pedagogy', '')}") # Fallback to raw string

        elif category == "scaffolding":
            text_parts.append(f"Learning Task: {row.get('Learning_Objective_Task', '')}")
            text_parts.append(f"Specific Student Struggle: {row.get('Specific_Struggle_Point', '')}")
            text_parts.append(f"Reasoning for choosing this scaffold: {row.get('reasoning_for_scaffold_choice', '')}")
            text_parts.append(f"Script to deliver the scaffold to the student: {row.get('scaffold_delivery_script', '')}")
            
        elif category == "feedback":
            text_parts.append(f"Student Proficiency and Task: {row.get('Proficiency', '')} {row.get('Task', '')}")
            text_parts.append(f"Observed Student Behavior: {row.get('Behavior Factor', '')}")
            text_parts.append(f"Specific Error Made by Student: {row.get('Error', '')}")
            text_parts.append(f"Expert Diagnosis of the Error: {row.get('Diagnose', '')}")
            text_parts.append(f"Strategy to Explain the Correction: {row.get('Explain Strategy', '')}")
            
        elif category == "teaching":
            text_parts.append(f"Learning Objective: {row.get('LEARNING_OBJECTIVE', '')}")
            text_parts.append(f"Core Teaching Strategy: {row.get('CORE_EXPLANATION_STRATEGY', '')}")
            text_parts.append(f"Key Examples Used: {row.get('KEY_EXAMPLES', '')}")
            text_parts.append(f"Common Misconceptions to Address: {row.get('COMMON_MISCONCEPTIONS', '')}")
            
        else:
            logging.warning(f"No specific embedding document creation logic for category: '{category}'. Using a generic fallback.")
            text_parts = [f"{k}: {v}" for k, v in row.iloc[:5].items()]

        return " \n ".join(filter(None, text_parts)).strip()
    except Exception as e:
        logging.error(f"Error creating document for category {category} on row {row.name}: {e}")
        return "" # Return empty string on error to avoid halting the whole process

def unify_and_standardize_cta_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, 'data', 'source_cta')
    output_dir = os.path.join(project_root, 'data', 'unified')
    output_path = os.path.join(output_dir, 'unified_knowledge_base.jsonl')
    os.makedirs(output_dir, exist_ok=True)

    source_files = {
        "cowriting_data.csv": "cowriting",
        "modelling_data.csv": "modelling",
        "pedagogical_data.csv": "pedagogy",
        "scaffolding_data.csv": "scaffolding",
        "teaching_data.csv": "teaching",
        "feedback_data.csv": "feedback",
    }
    
    final_standardized_records = []
    logging.info("Starting unification and standardization process...")

    for filename, category in source_files.items():
        file_path = os.path.join(source_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"File not found, skipping: {file_path}")
            continue
        try:
            logging.info(f"Processing '{filename}' for category '{category}'...")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip() # Strip whitespace from column names

            for _, row in df.iterrows():
                embedding_doc = create_embedding_document(row, category)
                if not embedding_doc:
                    logging.warning(f"Skipping row {row.name} in {filename} due to an error in document creation.")
                    continue
                
                original_row_dict = row.to_dict()
                metadata = {k.strip(): str(v) for k, v in original_row_dict.items()}
                
                # Default category is the filename's base category
                metadata['category'] = category

                # For 'teaching' files, overwrite the category with the specific learning objective ID
                if category == 'teaching':
                    learning_objective_full = str(row.get('LEARNING_OBJECTIVE', '')).strip()
                    if learning_objective_full:
                        # Extract the ID part (e.g., "SPK_GEN_FLU_001") before the colon
                        lesson_id = learning_objective_full.split(':')[0].strip()
                        if lesson_id:
                            metadata['category'] = lesson_id
                            # Also add lesson_id as a separate field for clarity
                            metadata['lesson_id'] = lesson_id
                
                standardized_record = {
                    "document_for_embedding": embedding_doc,
                    "metadata": metadata 
                }
                final_standardized_records.append(standardized_record)

            logging.info(f"Successfully processed and standardized {len(df)} rows from '{filename}'.")
        except Exception as e:
            logging.error(f"Failed to process file {filename}. Error: {e}", exc_info=True)
            continue
            
    if not final_standardized_records:
        logging.error("No records were processed. Unification failed.")
        return

    logging.info(f"Unification complete. Total standardized records: {len(final_standardized_records)}")
    try:
        logging.info(f"Saving unified knowledge base to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in final_standardized_records:
                f.write(json.dumps(record) + '\n')
        logging.info("Successfully saved the unified JSONL file.")
    except Exception as e:
        logging.error(f"Failed to save the unified JSONL file. Error: {e}", exc_info=True)

if __name__ == "__main__":
    unify_and_standardize_cta_data()