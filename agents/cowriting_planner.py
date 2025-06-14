
import logging
import os
import copy
from typing import Dict, Any, List
import json

from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

FALLBACK_COWRITING_PLAN = {
    "decision_to_intervene": True,
    "intervention_type": "SuggestRephrasing",
    "ai_spoken_or_suggested_text": "I notice you've been working on this section. Would you like me to help you refine your writing?",
    "original_student_text_if_revision": "",
    "suggested_ai_revision_if_any": "",
    "rationale_for_intervention_style": "Providing general support when specific planning was not possible",
    "anticipated_next_student_action_or_reply": "Student may accept or decline assistance",
    "ui_action_hints": [{"action_type_suggestion": "SHOW_SUGGESTION_BUTTON"}]
}

def cowriting_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering cowriting_planner_node")
    
    state_copy = copy.deepcopy(state)
    
    try:
        cowriting_strategies = state_copy.get("cowriting_strategies", [])
        student_written_chunk = state_copy.get("student_written_chunk", "")
        student_articulated_thought = state_copy.get("student_articulated_thought", "")
        student_affective_state = state_copy.get("student_affective_state", "")
        writing_task_context = state_copy.get("writing_task_context", {})
        task_type = writing_task_context.get("task_type", "")
        section = writing_task_context.get("section", "")
        student_comfort_level = state_copy.get("student_comfort_level", "")
        cowriting_lo_focus = state_copy.get("cowriting_lo_focus", "")
        writing_strengths = state_copy.get("writing_strengths", [])
        writing_weaknesses = state_copy.get("writing_weaknesses", [])
        
        logger.debug(f"Student written chunk: {student_written_chunk[:50]}...")
        logger.debug(f"Writing task: {task_type} - {section}")
        logger.debug(f"Student affective state: {student_affective_state}")
        logger.debug(f"Available strategies: {len(cowriting_strategies)}")
        
        if not cowriting_strategies:
            logger.warning("No cowriting strategies available for planning")
            state_copy["selected_cowriting_intervention"] = FALLBACK_COWRITING_PLAN
            return state_copy
        
        if not student_written_chunk:
            logger.warning("No student written chunk available")
            state_copy["selected_cowriting_intervention"] = FALLBACK_COWRITING_PLAN
            return state_copy
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    convert_system_message_to_human=True,
                    temperature=0.3,
                )
                
                strategies_text = json.dumps(cowriting_strategies, indent=2)
                
                prompt = f"""
                You are an expert AI writing coach planning the best cowriting intervention for a student.
                
                STUDENT CONTEXT:
                - Current Writing Task: {task_type}
                - Section: {section}
                - Learning Objective Focus: {cowriting_lo_focus or 'General writing improvement'}
                - Student's Comfort Level: {student_comfort_level or 'Unknown'}
                - Student's Current Affective State: {student_affective_state}
                
                STUDENT INPUT:
                - Written Text: "{student_written_chunk}"
                - Articulated Thought: "{student_articulated_thought or '[No articulated thought]'}"
                
                ANALYSIS:
                - Strengths: {', '.join(writing_strengths) if writing_strengths else 'Not analyzed'}
                - Weaknesses: {', '.join(writing_weaknesses) if writing_weaknesses else 'Not analyzed'}
                
                AVAILABLE STRATEGIES:
                {strategies_text}
                
                Please select the most appropriate cowriting intervention strategy from the list above OR create a new one that better fits this specific student's needs.
                
                Your response should be a JSON object with these fields:
                1. "decision_to_intervene": [true/false] - Whether to provide immediate intervention
                2. "intervention_type": String describing the type of intervention (e.g., "SuggestRephrasing", "OfferVocab")
                3. "ai_spoken_or_suggested_text": The exact text the AI should say to the student
                4. "original_student_text_if_revision": If suggesting a revision, the specific text to replace
                5. "suggested_ai_revision_if_any": If suggesting a revision, the replacement text
                6. "ui_action_hints": Array of UI actions that should be taken (e.g., highlighting text)
                7. "rationale_for_intervention_style": Explanation of why this approach fits the student's needs
                8. "anticipated_next_student_action_or_reply": Prediction of how student will respond
                
                Focus on being helpful, encouraging, and specific to this student's needs.
                """
                
                response = llm.invoke(prompt)
                response_text = response.content
                
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        selected_intervention = json.loads(json_text)
                    else:
                        raise ValueError("No valid JSON found in response")
                    
                    state_copy["selected_cowriting_intervention"] = selected_intervention
                    logger.info("Successfully selected and adapted cowriting intervention strategy")
                    
                except Exception as json_error:
                    logger.error(f"Error parsing strategy JSON: {str(json_error)}")
                    state_copy["selected_cowriting_intervention"] = cowriting_strategies[0]
            
            except Exception as e:
                logger.error(f"Error during cowriting strategy planning: {str(e)}")
                state_copy["selected_cowriting_intervention"] = cowriting_strategies[0]
        else:
            logger.warning("No API key available for cowriting planning, using first strategy")
            state_copy["selected_cowriting_intervention"] = cowriting_strategies[0]
        
        return state_copy
        
    except Exception as e:
        logger.error(f"Unexpected error in cowriting_planner_node: {str(e)}")
        state_copy["error"] = f"Cowriting planner error: {str(e)}"
        state_copy["selected_cowriting_intervention"] = FALLBACK_COWRITING_PLAN
        return state_copy
