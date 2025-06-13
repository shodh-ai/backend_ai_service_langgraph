import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def inactivity_prompt_node(state: AgentGraphState) -> Dict[str, Any]:
    logger.info(f"Inactivity Prompt Node activated for user {state.get('user_id', 'unknown_user')}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set for inactivity prompt node.")
        return {
            "output_content": {
                "text_for_tts": "I was about to check in, but I'm having a little technical difficulty.",
                "ui_actions": None
            },
            "student_memory_context": state.get("student_memory_context", {})
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )

        user_id = state.get("user_id", "student")
        current_context = state.get("current_context", {})
        student_memory = state.get("student_memory_context") or {}
        active_persona = state.get("active_persona", "Nurturer")
        chat_history = state.get("chat_history", [])

        student_name = student_memory.get("profile", {}).get("name", "Student")

        # Prepare student memory context for the prompt
        # Re-use the safely-loaded 'student_memory' variable.
        profile_data = student_memory.get("profile", {})
        interaction_history = student_memory.get("interaction_history", [])

        # Initialize/update user_id, student_name, task_stage from current_context
        current_context_obj = state.get("current_context")
        task_stage = "your current activity"  # Default for task_stage

        if current_context_obj:
            user_id = getattr(current_context_obj, "user_id", user_id)  # Fallback to existing user_id
            student_name = getattr(current_context_obj, "student_name", student_name)  # Fallback to existing student_name
            task_stage = getattr(current_context_obj, "task_stage", task_stage) # Fallback to default task_stage
            logger.info(f"inactivity_prompt_node: Fetched from current_context_obj - user_id: {user_id}, student_name: {student_name}, task_stage: {task_stage}")
        else:
            logger.warning("inactivity_prompt_node: current_context is None or not an object in state. Using default/existing values for user_id, student_name, and default for task_stage.")

        # Log the received context for debugging
        logger.debug(f"Inactivity Node - User ID: {user_id}, Name: {student_name}, Stage: {task_stage}")
        logger.debug(f"Inactivity Node - Profile: {profile_data}, History Count: {len(interaction_history)}")

        prompt = f"""
        You are Rox, an AI TOEFL Tutor, currently embodying the 'Nurturer' persona.
        The student (user_id: {user_id}, name: {student_name}) has been inactive for a little while during the task: '{task_stage}'.
        Your goal is to gently re-engage the student.
        Instructions:
        1. Acknowledge their current task/situation if relevant (from task_stage).
        2. Express a gentle check-in due to inactivity.
        3. Offer a simple way to continue or ask if they need help/a break.
        4. Maintain the 'Nurturer' persona (empathetic, encouraging, supportive).
        5. Return your response as a JSON object with the EXACT following structure:
           {{"text_for_tts": "Your re-engagement message...", "ui_actions": null}}
           (ui_actions should be null as per current spec for inactivity prompts)

        Example (Nurturer):
        {{"text_for_tts": "Hey {student_name}, just checking in. I noticed you've been quiet for a bit while working on {task_stage}. Everything alright? Need a moment, or some help to get going again?", "ui_actions": null}}
        """

        logger.debug(f"Inactivity Prompt LLM Prompt:\n{prompt}")
        response = await model.generate_content_async(prompt)
        logger.debug(f"Inactivity Prompt LLM Raw Response: {response.text}")
        response_json = json.loads(response.text)

        updated_student_memory = {
            **student_memory,
            "last_ai_action_on_p1": "sent_inactivity_prompt_waiting_for_reply"
        }

        return {
            "output_content": {
                "text_for_tts": response_json.get("text_for_tts", "Just checking in! Are you still there?"),
                "ui_actions": response_json.get("ui_actions") # Should be null as per spec
            },
            "student_memory_context": updated_student_memory
        }

    except Exception as e:
        logger.error(f"Error in inactivity_prompt_node: {e}", exc_info=True)
        return {
            "output_content": {
                "text_for_tts": "It seems I had a brief hiccup. Are you still with me?",
                "ui_actions": None
            },
            "student_memory_context": state.get("student_memory_context", {})
        }
