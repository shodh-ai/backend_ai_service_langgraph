# agents/rox_nodes.py
import logging
import json
from typing import Dict, Any
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from state import AgentGraphState
# Assume knowledge_client is available for fetching task titles etc.
from knowledge.knowledge_client import knowledge_client

logger = logging.getLogger(__name__)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

async def rox_welcome_and_reflection_node(state: AgentGraphState) -> dict:
    """
    The first "speaking" node on the Rox page. It adapts its message based on
    whether the user is brand new or returning from a task.
    """
    logger.info("--- Executing Rox Welcome & Reflection Node ---")
    
    task_stage = state.get("current_context", {}).get("task_stage")
    student_profile = state.get("student_memory_context", {})
    student_name = student_profile.get("name", "there")
    
    # This node uses an LLM to craft a natural, persona-aligned message
    prompt_instruction = ""
    if task_stage == "ROX_WELCOME_INIT":
        prompt_instruction = f"Craft a warm, initial welcome message for a student named '{student_name}'. Introduce yourself as Rox, their AI TOEFL Tutor. Briefly explain that your purpose is to guide them through a personalized learning path based on their goals and performance."
    elif task_stage == "ROX_RETURN_AFTER_TASK":
        last_task_context = state.get("current_context", {}).get("last_completed_task_details", {})
        task_title = last_task_context.get("title", "the last exercise")
        task_outcome = last_task_context.get("outcome_summary", "made some good progress")
        
        prompt_instruction = f"""
        Craft a reflective message for a student named '{student_name}' who has just returned to the home page after completing a task.
        - Acknowledge the completed task: '{task_title}'.
        - Mention their performance summary: '{task_outcome}'.
        - Be encouraging and set the stage for discussing what's next.
        Example tone: 'Welcome back! Great work on completing "{task_title}". The diagnosis showed you {task_outcome}. Let's see what the best next step is for you.'
        """
    else: # Fallback case
        prompt_instruction = f"Provide a generic, friendly greeting for {student_name} and ask what they would like to do."

    # Using an LLM for the text makes the greeting feel less robotic and can incorporate persona
    # model = genai.GenerativeModel("gemini-2.0-flash")
    # response = await model.generate_content_async(prompt_instruction)
    # greeting_text = response.text
    
    # Placeholder for Phase 1 without LLM call here:
    if task_stage == "ROX_RETURN_AFTER_TASK":
         greeting_text = f"Welcome back, {student_name}! Great work on completing your last task. Let's see what's next on our map."
    else:
         greeting_text = f"Hello {student_name}! I'm Rox, your AI TOEFL tutor. My job is to guide you on a personalized learning path. Let's start by looking at our curriculum map."
    
    # Per user guidance, explicitly pass the student_memory_context forward to prevent it from being dropped.
    return {
        "output_content": {"text_for_tts": greeting_text, "ui_actions": []},
        "student_memory_context": student_profile
    }


async def rox_propose_plan_node(state: AgentGraphState) -> dict:
    """
    Takes the output of the CurriculumNavigatorNode and presents the plan to the student.
    """
    logger.info("--- Executing Rox Propose Plan Node ---")
    
    # current_lo_to_address was set by curriculum_navigator_node
    next_lo_info = state.get("current_lo_to_address", {})
    lo_title = next_lo_info.get("title", "the next topic")
    
    # Here you could fetch the full LO Tree from knowledge base to explain the map
    # curriculum_map = await knowledge_client.get_full_curriculum_map()
    # map_explanation = "Our curriculum map shows that after mastering X, the next step is Y..." # LLM could generate this
    
    proposal_text = f"Based on your progress, our curriculum map suggests the best thing to focus on now is '{lo_title}'. This is a key skill for improving your overall writing score. Are you ready to start this lesson?"

    # The UI Action to display the plan/map would be generated here
    ui_actions = [{
        "action_type": "DISPLAY_CURRICULUM_MAP",
        "parameters": {
            "full_map_data": {}, # The full curriculum map object
            "current_lo_to_highlight": next_lo_info.get("id")
        }
    }]
    
    # Must pass the LO forward so the next node (the planner) can use it.
    return {
        "output_content": {"text_for_tts": proposal_text, "ui_actions": ui_actions},
        "last_action_was": "CURRICULUM_PROPOSAL", # Flag for saving this interaction
        "current_lo_to_address": next_lo_info
    }

async def rox_nlu_and_qa_handler_node(state: AgentGraphState) -> dict:
    """
    Handles NLU/Intent Classification for the student's turn on the Rox page.
    If the intent is a question, it also generates a response.
    """
    logger.info("--- Executing Rox NLU & QA Handler Node ---")
    
    updates = {}
    transcript = (state.get("transcript") or "").lower()

    # Phase 1 Stubbed NLU Logic
    confirmation_keywords = ["yes", "ok", "sure", "ready", "good", "alright", "sounds good", "let's do it", "confirm"]
    rejection_keywords = ["no", "different", "something else", "don't want to", "not ready"]

    if any(keyword in transcript for keyword in confirmation_keywords):
        student_intent = "CONFIRM_START_TASK"
    elif any(keyword in transcript for keyword in rejection_keywords):
        student_intent = "REQUEST_ALTERNATIVE_TASK"
    else: # Assume it's a question for now
        student_intent = "ASK_QUESTION"

    updates["student_intent_for_rox_turn"] = student_intent
    logger.info(f"Rox NLU: Classified intent as '{student_intent}' from transcript: '{transcript}'")

    # If it was a question, ALSO generate an answer
    if student_intent == "ASK_QUESTION":
        # In a real scenario, an LLM would generate this based on the transcript and context
        answer_text = "That's a great question! I've made a note of it. For now, to keep us on track, are you ready to start the lesson on the suggested topic?"
        updates["output_content"] = {"text_for_tts": answer_text, "ui_actions": []}
    
    # If the intent is NOT a question, this node produces no direct output,
    # it only sets the intent for the downstream router.
    
    return updates

async def rox_navigate_to_task_node(state: AgentGraphState) -> dict:
    """Prepares the state to navigate the user to their confirmed task."""
    logger.info("--- Executing Rox Navigate to Task Node ---")
    
    # The pedagogical_plan should have been created by a planner node by now
    plan = state.get("pedagogical_plan")
    if not plan:
        # Fallback if plan wasn't created - should not happen in happy path
        return {"output_content": {"text_for_tts": "Looks like there was an issue planning your next task. Let's try again."}}
        
    first_step = plan[0]
    modality = first_step.get("modality")
    target_page = first_step.get("target_page") # e.g., P7, P8
    details = first_step.get("details", {})
    
    response_text = f"Great! Taking you to your {modality} session on '{details.get('focus')}' now."

    # This UI action will trigger the frontend router
    ui_actions = [{
        "action_type": "NAVIGATE_TO_PAGE",
        "parameters": {
            "page_name": target_page,
            "data_for_page": details # Pass all details the new page will need
        }
    }]
    
    return {"output_content": {"text_for_tts": response_text, "ui_actions": ui_actions}}