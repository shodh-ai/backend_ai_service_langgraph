from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class InteractionRequestContext(BaseModel):
    user_id: str
    toefl_section: Optional[str] = None  # e.g., "Speaking", "Writing"
    question_type: Optional[str] = None  # e.g., "Q1_Independent", "Integrated_Essay"
    task_stage: Optional[str] = None  # e.g., "viewing_prompt", "prep_time", "active_response_speaking", "active_response_writing", "reviewing_feedback", "skill_drill_X"
    current_prompt_id: Optional[str] = None  # Unique ID for the current question/prompt
    ui_element_in_focus: Optional[str] = None  # e.g., "writing_paragraph_2", "speaking_point_1_notes"
    timer_value_seconds: Optional[int] = None  # If a timer is active
    selected_tools_or_options: Optional[Dict[str, Any]] = None  # e.g., {"highlighter_color": "yellow"}
    # ... any other relevant state from your UI ...


class InteractionRequest(BaseModel):
    transcript: Optional[str] = None
    current_context: Optional[InteractionRequestContext] = None
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None  # e.g. [{"role": "user", "content": "..."}, {"role": "ai", "content": "..."}]


class DomAction(BaseModel):
    action: str  # e.g., "update_text", "highlight_element", "play_audio"
    payload: Dict[str, Any]  # e.g., {"element_id": "paragraph1", "text": "New content"} or {"element_id": "sentence2", "color": "yellow"}


class InteractionResponse(BaseModel):
    response: str  # Text content intended for Text-to-Speech (TTS)
    dom_actions: Optional[List[DomAction]] = None # Optional list of actions for the UI to perform
