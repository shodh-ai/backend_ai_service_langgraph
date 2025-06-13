import logging
import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from state import AgentGraphState
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def motivational_support_node(state: AgentGraphState) -> Dict[str, Any]:
    logger.info(f"Motivational Support Node activated for user {state.get('user_id', 'unknown_user')}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set for motivational node.")
        # Fallback or error state
        return {
            "output_content": {
                "text_for_tts": "I'm here to help, but I'm having a little trouble accessing my full support features right now.",
                "ui_actions": []
            }
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash", # Using a capable model for nuanced responses
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )

        # Constructing the prompt based on your detailed request
        # Simplified for this example, you'd populate these from state
        user_id = state.get("user_id", "student")
        transcript = state.get("transcript", "")
        current_context = state.get("current_context")
        task_stage = getattr(current_context, 'task_stage', 'unknown_stage')
        
        student_memory = state.get("student_memory_context")
        if student_memory is None:
            student_memory = {}
        affective_state = student_memory.get("affective_state", "neutral")
        student_name = student_memory.get("name", "Student")
        active_persona = state.get("active_persona", "Nurturer") # Default to Nurturer if not set
        triggering_event = state.get("triggering_event_for_motivation", "general check-in")

        # This is a condensed version of your example prompt logic
        prompt_parts = [
            f"You are Rox, an AI TOEFL Tutor, currently embodying the '{active_persona}' persona.",
            f"The student (user_id: {user_id}) might be feeling '{affective_state}'. Their name from profile is {student_memory.get('profile', {}).get('name', 'Student')}.",
            f"This support is triggered by: '{triggering_event}'.",
            f"The student's last message (if relevant) was: '{transcript}'.",
            f"They are currently at task stage: '{task_stage}'.",
            f"Here is the student's profile data for context: {json.dumps(student_memory.get('profile', {}), indent=2)}",
            f"And here is the student's recent interaction history for context (e.g., to see recent struggles or successes): {json.dumps(student_memory.get('interaction_history', [])[-5:], indent=2)} (showing last 5 interactions)",
            "Your goal is to provide a brief, supportive, and motivational message that:",
            "1. Acknowledges and validates their current feeling (if negative), using their name if available from the profile.",
            "2. Normalizes the struggle if appropriate, possibly referencing general patterns or specific recent interactions from their history if relevant and helpful.",
            "3. Offers encouragement and positive reinforcement, potentially highlighting a past success from their interaction history if applicable.",
            "4. If appropriate for the persona and situation, suggests a very small, actionable coping strategy or a way to reframe the situation.",
            f"5. Maintains the '{active_persona}' persona's style.",
            "   - Structuralist: Logical, encouraging effort, focus on process, structured improvement, practically supportive.",
            "   - Nurturer: Empathetic, validating feelings, building self-belief, gentle guidance.",
            "6. Keep the message concise (1-3 sentences).",
            "7. Conclude by gently guiding them back to a productive next step or by asking a soft question.",
            "Return your response as a JSON object with the EXACT following structure: ",
            "{\"text_for_tts\": \"Your motivational message...\", \"ui_actions\": [], \"suggested_next_graph_node_hint\": \"<OPTIONAL_NODE_NAME_OR_LOGIC_HINT>\"}",
            "Example for suggested_next_graph_node_hint: NODE_PRACTICE_SELECTOR, NODE_REVIEW_CONCEPT, or even a specific sub-task like REVIEW_COHERENCE_NOTES."
        ]
        prompt = "\n".join(prompt_parts)

        logger.debug(f"Motivational Support Prompt:\n{prompt}")
        response = await model.generate_content_async(prompt) # Use async version
        
        logger.debug(f"Motivational Support LLM Raw Response: {response.text}")
        response_json = json.loads(response.text)

        return {
            "output_content": {
                "text_for_tts": response_json.get("text_for_tts", "I'm here to encourage you!"),
                "ui_actions": response_json.get("ui_actions", [])
            },
            "next_node_hint_from_motivation": response_json.get("suggested_next_graph_node_hint"),
            "student_memory_context": {**student_memory, "affective_state": f"acknowledged_{affective_state}"} # Update affective state
        }

    except Exception as e:
        logger.error(f"Error in motivational_support_node: {e}", exc_info=True)
        # Fallback message
        return {
            "output_content": {
                "text_for_tts": "It's okay to find things challenging. Keep trying your best!",
                "ui_actions": []
            },
            "next_node_hint_from_motivation": "NODE_CONVERSATION_HANDLER" # Default next step on error
        }
