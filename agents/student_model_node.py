import logging
import json
from datetime import datetime
from typing import Any, Dict

from state import AgentGraphState
from memory.mem0_client import shared_mem0_client

logger = logging.getLogger(__name__)

# --- Node Names ---
NODE_LOAD_STUDENT_DATA = "load_student_data"
NODE_SAVE_INTERACTION = "save_interaction"

async def load_student_data_node(state: AgentGraphState) -> dict:
    """
    Loads student data from Mem0, processes it into a structured profile and 
    interaction history, and extracts the last 5 interactions for recent context.
    """
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading student data for user_id: '{user_id}' from Mem0")

    student_data = {"profile": {}, "interaction_history": []}
    next_task_details = None

    try:
        all_memories = shared_mem0_client.get_all(user_id=user_id)
        memories_list = []

        # --- Safely Parse Mem0 Response ---
        if isinstance(all_memories, list):
            memories_list = all_memories
        elif isinstance(all_memories, dict) and 'memories' in all_memories:
            memories_list = all_memories['memories']
        elif all_memories:
            logger.warning(f"Unexpected format from Mem0 client: {type(all_memories)}. Treating as single item list.")
            memories_list = [all_memories]
        
        logger.info(f"Found {len(memories_list)} memory records from Mem0.")

        # --- Process Memories into Profile and History ---
        for mem in memories_list:
            try:
                if not (isinstance(mem, dict) and 'data' in mem and isinstance(mem['data'], dict)):
                    continue

                metadata = mem['data'].get('metadata', {})
                mem_type = metadata.get('type')
                messages = mem['data'].get('messages', [])
                if not messages:
                    continue

                content_str = messages[0].get('content', '{}')
                content_data = json.loads(content_str) if content_str.strip().startswith('{') else {'text': content_str}

                # Build Profile from specific memory types
                if mem_type == 'initial_report':
                    student_data["profile"].update(content_data)
                elif mem_type == 'diagnosis_summary':
                    student_data["profile"].setdefault('diagnosis', []).append(content_data)
                
                # Build Interaction History from all relevant memories
                student_data["interaction_history"].append({
                    'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                    'type': mem_type,
                    'last_action': metadata.get('last_action'),
                    'task_stage': metadata.get('task_stage'),
                    'content': content_data
                })
            except Exception as e:
                logger.error(f"Skipping malformed memory item: {e}")
                    # Handle profile data
                continue

        # Sort history chronologically (newest first)
        student_data["interaction_history"].sort(key=lambda x: x.get('timestamp', '0'), reverse=True)
        
        # Extract 'next_task_details' from the most recent interaction if available
        if student_data["interaction_history"]:
            last_interaction = student_data["interaction_history"][0]
            if last_interaction.get('last_action') == 'rox_navigate_to_task':
                next_task_details = last_interaction.get('content')
                logger.info(f"Found 'next_task_details' from the last interaction: {next_task_details}")

    except Exception as e:
        logger.error(f"CRITICAL FAILURE in load_student_data_node for user '{user_id}': {e}", exc_info=True)
        # Return a default state to prevent crashing the entire flow
        return {
            "student_memory_context": {},
            "next_task_details": None,
            "recent_interaction_history": []
        }

    # --- Finalize and Return State ---
    recent_history = student_data.get("interaction_history", [])[:5]
    logger.info(f"Extracted last {len(recent_history)} interactions for recent context.")

    return {
        "student_memory_context": student_data.get("profile", {}),
        "next_task_details": next_task_details,
        "recent_interaction_history": recent_history
    }

async def save_interaction_node(state: AgentGraphState) -> dict:
    """
    Saves a summary of the interaction to the student's long-term memory (Mem0)
    based on the 'last_action_was' field in the state.
    """
    last_action = state.get("last_action_was")

    # --- Guard: Only save if a meaningful action has occurred ---
    if not last_action:
        logger.info("Skipping interaction save: No 'last_action_was' set in state.")
        return {}

    user_id = state["user_id"]
    task_stage = state.get("current_context", {}).get("task_stage")
    logger.info(f"StudentModelNode: Saving interaction for user_id='{user_id}', last_action='{last_action}', task_stage='{task_stage}'")

    try:
        # --- Action-Specific Memory Saving ---
        data_to_save = None
        memory_type = "general_interaction"

        if last_action == "FEEDBACK_DELIVERY":
            data_to_save = state.get("intermediate_feedback_payload")
            memory_type = "feedback_session"
            logger.info("Preparing to save detailed feedback payload.")

        elif last_action in ["TEACHING_DELIVERY", "MODELLING_DELIVERY"]:
            data_to_save = state.get("output_content")
            memory_type = "instructional_delivery"
            logger.info(f"Preparing to save content from {last_action}.")
        
        elif last_action == "CURRICULUM_PROPOSAL":
            data_to_save = state.get("current_lo_to_address")
            memory_type = "learning_objective_proposal"
            logger.info("Preparing to save proposed learning objective.")

        elif last_action == "PEDAGOGY_PLANNER":
            data_to_save = state.get("pedagogical_plan")
            memory_type = "pedagogical_plan_creation"
            logger.info("Preparing to save the generated pedagogical plan.")

        # --- Save the structured data if any was found ---
        if data_to_save:
            content_to_save = json.dumps(data_to_save, indent=2) if isinstance(data_to_save, (dict, list)) else str(data_to_save)
            shared_mem0_client.add(
                messages=[{"role": "system", "content": content_to_save}],
                user_id=user_id,
                metadata={'type': memory_type, 'last_action': last_action, 'task_stage': task_stage, 'timestamp': datetime.now().isoformat()}
            )
            logger.info(f"Successfully saved structured memory of type '{memory_type}'.")

        # --- Always save the conversational turn as well for context ---
        transcript = state.get("transcript", "")
        output_content = state.get("output_content", {})
        assistant_text = output_content.get("text_for_tts", "") if isinstance(output_content, dict) else ""

        messages_to_save = []
        if transcript and transcript.strip():
            messages_to_save.append({"role": "user", "content": transcript})
        if assistant_text.strip():
            messages_to_save.append({"role": "assistant", "content": assistant_text})

        if messages_to_save:
            shared_mem0_client.add(
                messages=messages_to_save,
                user_id=user_id,
                metadata={'type': 'conversation_turn', 'last_action': last_action, 'task_stage': task_stage, 'timestamp': datetime.now().isoformat()}
            )
            logger.info(f"Successfully saved conversational turn for action '{last_action}'.")
        else:
            logger.warning(f"No conversational content to save for action '{last_action}'.")

    except Exception as e:
        logger.error(f"Failed to save interaction to Mem0: {e}", exc_info=True)

    # --- Clear the flag after processing to prevent re-saving in the next turn ---
    return {"last_action_was": None}
