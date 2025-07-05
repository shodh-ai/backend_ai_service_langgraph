import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from typing import Dict
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

async def conversation_handler_node(state: AgentGraphState) -> dict:
    """
    This node performs NLU/Intent Classification on the student's input.
    Its SOLE JOB is to understand what the student wants and update the state
    with that intent, so downstream routers can make decisions.
    """
    logger.info("---Executing Conversation Handler Node (Context-Aware NLU) ---")
    
    try:
        transcript = state.get("transcript", "")
        current_context = state.get("current_context", {})
        task_stage = current_context.get("task_stage", "")
        chat_history = state.get("chat_history", [])
        
        if not transcript.strip():
            logger.warning("ConversationHandlerNode: Transcript is empty. Nothing to process.")
            return {
                "classified_student_intent": "NO_INPUT",
                "classified_student_entities": None
            }

        # --- DYNAMIC PROMPT SELECTION BASED ON CONTEXT ---
        possible_intents = []
        nlu_instructions = ""

        if task_stage == "ROX_CONVERSATION_TURN":
            possible_intents = ["CONFIRM_PROCEED_WITH_LO", "REJECT_OR_QUESTION_LO", "REQUEST_STATUS_DETAIL", "GENERAL_CHITCHAT"]
            nlu_instructions = "The AI just proposed a new learning objective. Classify the student's response."
        
        elif task_stage in ["TEACHING_PAGE_QA", "MODELING_PAGE_QA", "SCAFFOLDING_PAGE_QA"]:
            possible_intents = ["ASK_CLARIFICATION_QUESTION", "REQUEST_EXAMPLE", "STATE_CONFUSION", "OFF_TOPIC_QUESTION", "CONTINUE_LESSON"]
            last_ai_statement = chat_history[-2]['content'] if len(chat_history) > 1 else "the current topic"
            nlu_instructions = f"The AI is in the middle of a lesson and just explained a concept. The AI's last statement was: '{last_ai_statement}'. Now, classify the student's following question/statement."

        else:
            possible_intents = ["GENERAL_CONFIRM", "GENERAL_REJECT", "GENERAL_QUESTION"]
            nlu_instructions = "Classify the user's general intent."

        prompt = f"""
        You are an NLU assistant for an AI Tutor.
        Context: {nlu_instructions}
        Student said: "{transcript}"

        Categorize the student's intent into ONE of the following types: {possible_intents}
        If the intent is a question, also extract the core topic of the question.

        Return ONLY a single valid JSON object with the format: {{"intent": "<INTENT_NAME>", "extracted_topic": "<topic if any, otherwise null>"}}
        """
        
        # --- Call the LLM ---
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(prompt)
        llm_response_json = json.loads(response.text)
        
        classified_intent = llm_response_json.get("intent")
        extracted_entities = llm_response_json.get("entities") or llm_response_json.get("extracted_topic")

        logger.info(f"NLU classified intent as: {classified_intent}")

        # --- THE CRITICAL FIX: Return a dictionary to update the state ---
        return {
            "classified_student_intent": classified_intent,
            "classified_student_entities": extracted_entities
        }
    except Exception as e:
        logger.error(f"ConversationHandlerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}
