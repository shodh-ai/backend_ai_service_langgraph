# agents/feedback_output_formatter.py
import logging
from state import AgentGraphState
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def create_ui_action(action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to create a UI action dictionary."""
    return {"action_type": action_type, "parameters": parameters}

async def feedback_output_formatter_node(state: AgentGraphState) -> dict:
    """
    Translates the generator's feedback plan into a choreographed sequence of UI actions.
    """
    logger.info("---Executing Feedback Output Formatter---")

    payload = state.get("intermediate_feedback_payload")
    if not payload or payload.get("error"):
        error_msg = payload.get("error_message", "An error occurred while preparing feedback.")
        logger.warning(f"Feedback Formatter received an error: {error_msg}")
        return {"final_text_for_tts": error_msg, "final_ui_actions": []}

    spoken_script = payload.get("spoken_script", [])
    feedback_items = payload.get("feedback_items", [])
    
    # The final TTS will be the entire spoken script joined together.
    final_tts = " ".join(filter(None, spoken_script))
    
    final_ui_actions: List[Dict[str, Any]] = []

    # --- Translate the feedback items into specific UI actions ---

    # 1. Create all the highlights at once.
    all_highlights = []
    for i, item in enumerate(feedback_items):
        highlight_data = item.get("highlight", {})
        remark_data = item.get("remark", {})
        if highlight_data and remark_data:
            all_highlights.append({
                "start": highlight_data.get("start"),
                "end": highlight_data.get("end"),
                "style_class": highlight_data.get("style_class"),
                "remark_id": remark_data.get("id", f"R{i+1}") # Use the linked ID
            })
    
    if all_highlights:
        final_ui_actions.append(create_ui_action(
            "HIGHLIGHT_TEXT_RANGES",
            {"targetElementId": "p6StudentWorkDisplay", "ranges": all_highlights}
        ))

    # 2. Create the list of remark cards to display in the panel.
    all_remarks = [item.get("remark") for item in feedback_items if item.get("remark")]
    if all_remarks:
        final_ui_actions.append(create_ui_action(
            "DISPLAY_REMARKS_LIST",
            {"targetElementId": "p6FeedbackRemarksPanel", "remarks": all_remarks}
        ))
        
    # 3. Add the spoken script to the chat bubble.
    # We can join it for a single update, or create a sequence for a live effect.
    # For now, let's do a single update.
    if final_tts:
        final_ui_actions.append(create_ui_action(
            "UPDATE_TEXT_CONTENT",
            {"targetElementId": "p6AiChatBubble", "text": final_tts}
        ))
    
    # 4. Set the initial focus title.
    initial_focus_title = feedback_items[0].get("remark", {}).get("title") if feedback_items else "Overall Feedback"
    final_ui_actions.append(create_ui_action(
        "UPDATE_TEXT_CONTENT",
        {"targetElementId": "p6CurrentFeedbackFocusTitle", "text": f"Let's discuss: {initial_focus_title}"}
    ))

    logger.info(f"Feedback formatter created TTS and {len(final_ui_actions)} UI actions.")

    # Standardize the output to a single dictionary for consistency across flows
    output_content = {
        "text_for_tts": final_tts,
        "ui_actions": final_ui_actions
    }

    return {
        "output_content": output_content,
        "last_action_was": "FEEDBACK_DELIVERY"
    }