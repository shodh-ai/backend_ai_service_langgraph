import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

async def prepare_navigation_node(state: AgentGraphState) -> Dict[str, Any]:
    user_id = state.get("user_id", "unknown_user")
    logger.info(f"Prepare Navigation Node activated for user {user_id}")

    next_task_details = state.get("next_task_details")
    if not next_task_details or not isinstance(next_task_details, dict):
        logger.error(f"'next_task_details' is missing or invalid in state for user {user_id}. Cannot prepare navigation.")
        # Fallback if next_task_details are missing
        # OutputFormatterNode will use default TTS if navigation_tts is not set
        return {
            "navigation_tts": "I seem to have lost track of what we were about to do. Could you remind me, or would you like to see other options?",
            "ui_actions_for_formatter": [], # Or suggest going back to main menu/conversation handler
            "next_node_override": "NODE_CONVERSATION_HANDLER" # Suggest routing back
        }

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set for prepare_navigation_node.")
        # Fallback to templated response if LLM is not available
        task_title_fallback = next_task_details.get("title", "your next activity")
        tts_text = f"Alright! Let's get started with {task_title_fallback}."
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                "gemini-2.0-flash",
                generation_config=GenerationConfig(response_mime_type="application/json"),
            )

            # Get comprehensive student memory context
            student_memory = state.get("student_memory_context")
            
            # Add null checking to prevent NoneType errors
            if student_memory is None:
                student_memory = {}
                logger.warning("Student memory context is None, using empty dictionary for navigation")
                
            profile = student_memory.get("profile", {})
            student_name = profile.get("name", "there") # Default to 'there' if name not found
            interaction_history = student_memory.get("interaction_history", [])
            active_persona = state.get("active_persona", "Nurturer")
            
            # Extract task details
            task_title = next_task_details.get("title", "the selected task")
            task_type = next_task_details.get("type", "activity")

            # Get student level and preferences if available
            student_level = profile.get("level", "")
            student_preferences = profile.get("preferences", {})
            student_focus_areas = profile.get("focus_areas", [])
            
            # Extract recent progress or relevant context from interaction history
            recent_task_info = ""
            if interaction_history and isinstance(interaction_history, list) and len(interaction_history) > 0:
                try:
                    # Get most recent interaction
                    last_interaction = interaction_history[-1]
                    if isinstance(last_interaction, dict):
                        # If there was a previous task, include information about it
                        last_task = last_interaction.get("task_details", {})
                        if last_task and isinstance(last_task, dict):
                            last_title = last_task.get("title", "")
                            if last_title and last_title != task_title:
                                recent_task_info = f"They just completed '{last_title}'."
                except Exception as e:
                    logger.warning(f"Error extracting recent task info: {e}")

            # Build enriched prompt with memory context
            prompt_parts = [
                f"You are Rox, an AI TOEFL Tutor, speaking as the '{active_persona}' persona.",
                f"The student, {student_name}, has just confirmed they are ready to start the following task/lesson:",
                f"Task Title: '{task_title}'",
                f"Task Type: '{task_type}'"
            ]
            
            # Add relevant profile information
            if student_level:
                prompt_parts.append(f"Student Level: {student_level}")
            
            if student_focus_areas and isinstance(student_focus_areas, list) and len(student_focus_areas) > 0:
                prompt_parts.append(f"Student Focus Areas: {', '.join(student_focus_areas)}")
                
            # Add recent task context if available
            if recent_task_info:
                prompt_parts.append(recent_task_info)
                
            # Instructions for response
            prompt_parts.extend([
                "Generate a brief, encouraging, and clear transitional phrase to say to the student as you navigate them to this task.",
                "The transition should feel natural and personalized based on their profile and history.",
                "Keep it concise (1 sentence).",
                "Return JSON: {\"text_for_tts\": \"<your transitional phrase>\"}"
            ])
            
            # Process the prompt and get response
            prompt = "\n".join(prompt_parts)
            logger.debug(f"Prepare Navigation LLM Prompt:\n{prompt}")
            response = await model.generate_content_async(prompt)
            response_json = json.loads(response.text)
            tts_text = response_json.get("text_for_tts", f"Okay, {student_name}, let's move on to {task_title}!")
        except Exception as e:
            logger.error(f"Error in prepare_navigation_node LLM processing: {e}", exc_info=True)
            # Fallback to templated response if LLM initialization or processing fails
            task_title_fallback = next_task_details.get("title", "your next activity")
            student_name_fallback = state.get("student_memory_context", {}).get("profile", {}).get("name", "there") 
            tts_text = f"Great, {student_name_fallback}! Let's move on to {task_title_fallback}."

    # Construct NAVIGATE_TO_PAGE UI Action
    page_target = next_task_details.get("page_target")
    if not page_target:
        logger.error(f"'page_target' is missing in next_task_details for user {user_id}. Cannot create NAVIGATE_TO_PAGE action.")
        # Fallback if page_target is missing
        return {
            "navigation_tts": "I'm ready to move on, but I'm not sure where we're going! Let's try choosing an activity again.",
            "ui_actions_for_formatter": [],
            "next_node_override": "NODE_CONVERSATION_HANDLER"
        }

    # Prepare data_for_page, filtering out None values
    data_for_page_raw = {
        "prompt_id": next_task_details.get("prompt_id"),
        "lesson_id": next_task_details.get("lesson_id"),
        "title": next_task_details.get("title"),
        "prep_time_seconds": next_task_details.get("prep_time"), # Assuming 'prep_time' is in seconds
        "speak_time_seconds": next_task_details.get("speak_time"), # Assuming 'speak_time' is in seconds
        "question_type": next_task_details.get("question_type"),
        # Add any other relevant details from next_task_details that the target page might need
    }
    data_for_page_cleaned = {k: v for k, v in data_for_page_raw.items() if v is not None}

    nav_ui_action = {
        "action_type": "NAVIGATE_TO_PAGE",
        "parameters": {
            "page_name": page_target,
            "data_for_page": data_for_page_cleaned
        }
    }

    # Optionally update student memory
    updated_student_memory = state.get("student_memory_context", {})
    
    if isinstance(updated_student_memory, dict):
      updated_student_memory["last_ai_action_on_p1"] = f"navigated_to_task_{next_task_details.get('title', 'unknown')}"

    return {
        "conversational_tts": tts_text,
        "ui_actions_for_formatter": [nav_ui_action],
        "student_memory_context": updated_student_memory
    }
