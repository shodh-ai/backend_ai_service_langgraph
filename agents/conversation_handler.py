import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from typing import Dict
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

async def conversation_handler_node(state: AgentGraphState) -> dict:
    logger.info("---Executing Conversation Handler Node (NLU Intent Classification)---")
    
    try:
        transcript = state.get("transcript")
        if not transcript or not transcript.strip():
            logger.warning("ConversationHandlerNode: Transcript is empty. Nothing to process.")
            return {"classified_intent": "NO_INPUT"}

        # Check for the last AI action to provide conversational context
        last_ai_action = state.get("last_ai_action")

        if last_ai_action == "PROPOSED_NEW_LO":
            # This is the specific context for handling the LO proposal
            lo_title = state.get("current_lo_to_address", {}).get("title", "the proposed topic")
            llm_prompt = f"""
As an NLU engine, your task is to classify the user's intent.
The AI just proposed working on "{lo_title}".
The user responded: "{transcript}"

Is the student's intent 'CONFIRM_PROCEED_WITH_LO' or 'REJECT_OR_QUESTION_LO'?
- 'CONFIRM_PROCEED_WITH_LO' means they agree, say yes, okay, etc.
- 'REJECT_OR_QUESTION_LO' means they say no, ask a question, or want to do something else.

Respond with a SINGLE JSON object with one key, "intent", and the classified intent as the value.
Example: {{"intent": "CONFIRM_PROCEED_WITH_LO"}}
"""
        elif state.get("interruption_context", {}).get("expected_intent"):
            # Focused prompt using the micro-intent
            llm_prompt = f"""
As an NLU engine, your task is to classify the user's intent.
The system was expecting the user to provide a response with the intent: '{expected_intent}'.
The user said: "{transcript}"

Analyze the user's response.
- If the user's intent matches the expected intent, respond with a JSON object: {{"intent": "{expected_intent}"}}
- If the user's intent is different, classify the new intent and respond with a JSON object: {{"intent": "NEW_INTENT_CLASSIFICATION"}}
- Possible intents include: CONFIRMATION, REQUEST_CLARIFICATION, DISAGREEMENT, QUESTION, PROVIDE_EXAMPLE, OFF_TOPIC.
"""
        else:
            # General-purpose intent classification
            llm_prompt = f"""
As an NLU engine, your task is to classify the user's intent based on their statement.
The user said: "{transcript}"

Classify the user's intent from the following list:
- REQUEST_TEACHING_LESSON
- START_MODELLING_ACTIVITY
- ASK_QUESTION
- PROVIDE_FEEDBACK
- GENERAL_CONVERSATION

Respond with a SINGLE JSON object with one key, "intent", and the classified intent as the value.
Example: {{"intent": "ASK_QUESTION"}}
"""

        logger.debug(f"NLU Prompt:\n{llm_prompt}")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        response = await model.generate_content_async(llm_prompt)
        response_json = json.loads(response.text)
        classified_intent = response_json.get("intent")
        
        logger.info(f"NLU classified intent as: {classified_intent}")

        return {"classified_intent": classified_intent}

    except Exception as e:
        logger.error(f"ConversationHandlerNode: CRITICAL FAILURE: {e}", exc_info=True)
        return {"error_message": str(e), "route_to_error_handler": True}
