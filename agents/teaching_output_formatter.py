# In your LangGraph project's teaching_output_formatter.py

import logging
import uuid
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def teaching_output_formatter_node(state: dict) -> dict:
    """
    Translates the generator's layered content plan into a single,
    robust SPEAK_THEN_LISTEN action for the livekit-service.
    """
    logger.info("---Executing Teaching Output Formatter---")
    
    # Primary path: generator saved a structured payload
    payload = state.get("intermediate_teaching_payload", {})

    # Fallback: delivery generator wrote its output directly on the state
    if not payload:
        if state.get("output_content"):
            payload = state.get("output_content")
        elif state.get("text_for_tts") is not None:
            payload = {
                "text_for_tts": state.get("text_for_tts"),
                "ui_actions": state.get("ui_actions", []),
            }
        else:
            logger.error("Formatter received an empty teaching payload and found no fallback keys.")
            return {"final_ui_actions": []}

    # If payload already matches final schema, just return it
    if payload and "text_for_tts" in payload and "ui_actions" in payload:
        text = payload.get("text_for_tts")
        ui_actions = payload.get("ui_actions") or []
        # If no UI actions, create a default SPEAK_THEN_LISTEN so the agent listens
        if not ui_actions:
            ui_actions.append({
                "action_type": "SPEAK_THEN_LISTEN",
                "speak_text": text,
                "listen_config": {"silence_timeout_ms": 3000}
            })
        return {
            "text_for_tts": text,
            "final_ui_actions": ui_actions,
        }

    sequence = payload.get("sequence", [])
    if not sequence:
        logger.warning("Generator's payload contained no 'sequence'.")
        return {"final_ui_actions": []}

    # --- THIS IS THE CRITICAL LOGIC ---
    
    # 1. Combine all TTS steps into a single block of text for natural speech flow.
    tts_parts = [step.get("content", "") for step in sequence if step.get("type") == "tts"]
    full_text_to_speak = " ".join(filter(None, tts_parts)).strip()

    # 2. Find the final 'listen' step to get its configuration.
    listen_step = next((step for step in reversed(sequence) if step.get("type") == "listen"), None)

    if not listen_step:
        logger.warning("Sequence did not end with a 'listen' step. Cannot create interactive action.")
        # If there's no listen, you could just send a SPEAK_TEXT action.
        # But for this flow, we expect one.
        return {"final_ui_actions": []}

    # 3. Build the single, powerful SPEAK_THEN_LISTEN action.
    speak_then_listen_action = {
        "action_type": "SPEAK_THEN_LISTEN",
        "parameters": {
            "speech_id": f"teach-{uuid.uuid4().hex[:8]}", # Give it a unique ID
            "text_to_speak": full_text_to_speak,
            "listen_config": {
                "timeout_ms": listen_step.get("timeout_ms", 8000),
                "prompt_if_silent": listen_step.get("prompt_if_silent", "Are you still there?"),
                "expected_intent": listen_step.get("expected_intent", "user_response")
            }
        }
    }
    
    # You can also add other, non-blocking UI actions here if needed.
    # For example, showing text on the screen.
    final_actions = [speak_then_listen_action]

    logger.info(f"Formatted plan into a single SPEAK_THEN_LISTEN action.")

    # Return in format expected by LiveKit service
    return {
        "text_for_tts": full_text_to_speak,
        "final_ui_actions": final_actions,
    }