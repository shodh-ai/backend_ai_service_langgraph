import logging
from state import AgentGraphState
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def semantic_search_for_scaffolding(
    data_entries: List[Dict[str, Any]], 
    query: str, 
    learning_objective_id: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY environment variable is not set - using fallback matching")
            return keyword_based_fallback_search(data_entries, query, learning_objective_id, top_k)
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        if learning_objective_id:
            objective_keywords = {
                "S_Q1_Structure": ["structure", "organization", "toefl speaking question 1"],
                "S_Q1_Fluency": ["fluency", "toefl speaking question 1"],
                "S_Q2_Coherence": ["coherence", "cohesion", "toefl speaking question 2"]
            }
            
            search_terms = objective_keywords.get(learning_objective_id, [learning_objective_id])
            
            filtered_entries = []
            for entry in data_entries:
                objective_text = entry.get("Learning_Objective_Task", "").lower()
                if any(term.lower() in objective_text for term in search_terms):
                    filtered_entries.append(entry)
                    
            if not filtered_entries and data_entries:
                logger.warning(f"No entries found for learning objective {learning_objective_id} with search terms {search_terms}. Using all entries.")
                filtered_entries = data_entries
        else:
            filtered_entries = data_entries
            
        logger.info(f"Filtered to {len(filtered_entries)} entries for learning objective {learning_objective_id}")
        
        
        if not filtered_entries:
            return []
            
        texts = [
            f"{entry.get('Specific_Struggle_Point', '')} {entry.get('Learning_Objective_Task', '')}"
            for entry in filtered_entries
        ]
        
        query_embedding = embeddings.embed_query(query)
        
        text_embeddings = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings.embed_documents(batch_texts)
            text_embeddings.extend(batch_embeddings)
        
        similarities = []
        for text_embedding in text_embeddings:
            similarity = np.dot(query_embedding, text_embedding) / \
                         (np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding))
            similarities.append(similarity)
        
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [filtered_entries[i] for i in top_k_indices]
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        logger.info("Falling back to keyword-based matching")
        return keyword_based_fallback_search(data_entries, query, learning_objective_id, top_k)


def keyword_based_fallback_search(
    data_entries: List[Dict[str, Any]],
    query: str,
    learning_objective_id: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Fallback search method using simple keyword matching when embeddings aren't available.
    """
    logger.info("Using keyword-based fallback search")
    
    keywords = query.lower().split()
    keywords = [k for k in keywords if len(k) > 3]
    
    if learning_objective_id:
        filtered_entries = [entry for entry in data_entries 
                         if learning_objective_id in entry.get("Learning_Objective_Task", "")]
    else:
        filtered_entries = data_entries
        
    scored_entries = []
    for entry in filtered_entries:
        score = 0
        struggle_text = entry.get('Specific_Struggle_Point', '').lower()
        objective_text = entry.get('Learning_Objective_Task', '').lower()
        
        for keyword in keywords:
            if keyword in struggle_text:
                score += 2
            if keyword in objective_text:
                score += 1
                
        scored_entries.append((entry, score))
    
    scored_entries.sort(key=lambda x: x[1], reverse=True)
    return [entry for entry, score in scored_entries[:top_k]]


async def scaffolding_retriever_node(state: AgentGraphState) -> dict:
    """
    Retrieves relevant scaffolding strategies based on student struggles and learning objectives.
    
    This node uses semantic search to find appropriate scaffolding strategies from
    a dataset of pre-generated scaffolding approaches.
    """
    logger.info(
        f"ScaffoldingRetrieverNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    
    primary_struggle = state.get("primary_struggle", "")
    secondary_struggles = state.get("secondary_struggles", [])
    learning_objective_id = state.get("learning_objective_id", "")
    
    sample_scaffolding_data = [
        {
            "Persona": "The Structuralist",
            "Learning_Objective_Task": "S_Q1_Structure: Structuring a Response for TOEFL Speaking Question 1",
            "Specific_Struggle_Point": "Student gives an opinion but then rambles without clearly distinct reasons or examples.",
            "scaffold_type_selected": "Opinion Response Framework Template",
            "scaffold_content_delivered_type": "template",
            "scaffold_content_delivered_name": "OREO Speaking Template",
            "scaffold_content_delivered_content_json": '{"fields": [{"label": "Opinion:", "placeholder": "I believe that..."}, {"label": "Reason 1:", "placeholder": "One reason is..."}, {"label": "Example 1:", "placeholder": "For example..."}, {"label": "Reason 2:", "placeholder": "Another reason is..."}, {"label": "Example 2 (optional):", "placeholder": "For instance..."}, {"label": "Opinion restated:", "placeholder": "That\'s why I think..."}]}'
        },
        {
            "Persona": "The Encouraging Nurturer",
            "Learning_Objective_Task": "W_Ind_Thesis: Writing a Clear Thesis Statement for TOEFL Independent Essay",
            "Specific_Struggle_Point": "Student has written a topic but not an arguable thesis. They say 'I don't know what my main argument is.'",
            "scaffold_type_selected": "Thesis Statement Builder",
            "scaffold_content_delivered_type": "template",
            "scaffold_content_delivered_name": "Position-Reason Thesis Template",
            "scaffold_content_delivered_content_json": '{"fields": [{"label": "Topic:", "placeholder": "e.g., technology in education"}, {"label": "Position:", "placeholder": "I believe technology in education is beneficial/harmful"}, {"label": "Main Reason 1:", "placeholder": "because it increases engagement"}, {"label": "Main Reason 2:", "placeholder": "and provides access to more resources"}]}'
        },
        {
            "Persona": "The Structuralist",
            "Learning_Objective_Task": "S_Q1_Fluency: Speaking Fluently for TOEFL Speaking Question 1",
            "Specific_Struggle_Point": "Student starts speaking but hesitates frequently after the first few words, using many filler words like 'um' and 'uh'.",
            "scaffold_type_selected": "Sentence Starter Bank",
            "scaffold_content_delivered_type": "sentence_starters",
            "scaffold_content_delivered_name": "Opinion & Reasoning Starters",
            "scaffold_content_delivered_content_json": '{"starters": ["In my opinion...", "I believe that...", "From my perspective...", "One important reason is...", "A key factor to consider is...", "For example, in my experience...", "To illustrate this point...", "This can be seen when..."]}'
        }
    ]
    
    is_test = os.environ.get('SCAFFOLDING_TEST_MODE', 'false').lower() == 'true'
    
    if is_test:
        csv_filename = "scaffolding_test_data.csv"
        logger.info("Using smaller TEST CSV dataset for scaffolding retrieval")
    else:
        csv_filename = "scaffolding_cta_data.csv"
    
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        csv_filename
    )
    
    try:
        scaffolding_data = pd.read_csv(csv_path).to_dict('records')
        logger.info(f"Successfully loaded scaffolding data from {csv_path}")
    except Exception as e:
        logger.error(f"Error loading scaffolding data: {e}")
        logger.info("Falling back to sample scaffolding data")
        scaffolding_data = sample_scaffolding_data
    
    struggle_description = primary_struggle
    if secondary_struggles:
        struggle_description += ". " + " ".join(secondary_struggles)
        
    logger.info(f"Using struggle description for search: '{struggle_description}'")
    logger.info(f"Using learning objective ID for filtering: '{learning_objective_id}'")
    
    relevant_strategies = semantic_search_for_scaffolding(
        data_entries=scaffolding_data,
        query=struggle_description,
        learning_objective_id=learning_objective_id
    )
    
    logger.info(f"Retrieved {len(relevant_strategies)} relevant scaffolding strategies")
    
    new_state = {key: value for key, value in state.items()}
    
    new_state["scaffolding_strategies"] = relevant_strategies
    
    logger.info(f"Retriever returning {len(relevant_strategies)} strategies") 
    logger.info(f"Returning state with keys: {list(new_state.keys())}")
    
    return new_state
