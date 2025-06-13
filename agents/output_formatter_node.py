import logging
from typing import Dict, Any, List, Optional
from state import AgentGraphState

logger = logging.getLogger(__name__)

async def format_final_output_for_client_node(state: AgentGraphState) -> Dict[str, Any]:
    """
    Consolidates all intended outputs for the client (text for TTS, UI actions, 
    navigation instructions, next task information, raw system outputs) from 
    various fields of the AgentGraphState populated by preceding agent nodes.
    Structures this information into a definitive dictionary for the FastAPI endpoint.
    
    Args:
        state: The current agent graph state
        
    Returns:
        A dictionary containing the final structured output for the client.
    """
    node_name = "OutputFormatterNode" # Consistent logging prefix
    logger.info(f"{node_name}: Formatting final output for the client.")

    client_response_data: Dict[str, Any] = {}

    # --- Retrieve and Consolidate Conversational TTS ---
    conversational_tts = ""
    current_context = state.get("current_context")
    task_stage = ""
    if current_context:
        task_stage = getattr(current_context, "task_stage", "").upper()
    
    modelling_output_content = state.get("modelling_output_content") or {}
    teaching_output = state.get("teaching_output_content") or {}

    # Prioritize modelling TTS if in a modelling stage
    if "MODELLING" in task_stage:
        logger.info(f"{node_name}: Task stage '{task_stage}' indicates modelling. Prioritizing 'modelling_output_content' for TTS.")
        if isinstance(modelling_output_content, dict) and modelling_output_content.get("text_for_tts"):
            tts_candidate = str(modelling_output_content["text_for_tts"])
            if tts_candidate.strip():
                conversational_tts = tts_candidate
                logger.info(f"{node_name}: Using 'text_for_tts' from 'modelling_output_content' (task stage priority): '{conversational_tts[:100]}...' ")

    # Check for PEDAGOGY_GENERATION stage if modelling TTS wasn't used or not in modelling stage
    if not conversational_tts and task_stage == "PEDAGOGY_GENERATION":
        logger.info(f"{node_name}: Task stage is 'PEDAGOGY_GENERATION'. Checking 'task_suggestion_llm_output' for TTS.")
        pedagogy_output = state.get("task_suggestion_llm_output") or {}
        if isinstance(pedagogy_output, dict):
            tts_candidate_pedagogy = pedagogy_output.get("task_suggestion_tts")
            if tts_candidate_pedagogy and isinstance(tts_candidate_pedagogy, str) and tts_candidate_pedagogy.strip():
                conversational_tts = tts_candidate_pedagogy
                logger.info(f"{node_name}: Using 'task_suggestion_tts' from 'task_suggestion_llm_output': '{conversational_tts[:100]}...' ")
            else:
                logger.info(f"{node_name}: 'task_suggestion_tts' from 'task_suggestion_llm_output' is empty, not a string, or not found.")
        else:
            logger.info(f"{node_name}: 'task_suggestion_llm_output' is not a dictionary or not found.")

    # Fallback to teaching_output if modelling/pedagogy TTS wasn't used or not in a relevant stage and conversational_tts is still empty
    if not conversational_tts:
        logger.info(f"{node_name}: Checking 'teaching_output_content' for TTS (task_stage: '{task_stage}'): {teaching_output.get('text_for_tts') is not None}")
        if isinstance(teaching_output, dict) and teaching_output.get("text_for_tts"):
            tts_candidate = str(teaching_output["text_for_tts"])
            if tts_candidate.strip():
                conversational_tts = tts_candidate
                logger.info(f"{node_name}: Using 'text_for_tts' from 'teaching_output_content': '{conversational_tts[:100]}...' ")
    
    # Further fallbacks (original logic)
    if not conversational_tts:
        raw_conversational_tts_direct = state.get("conversational_tts")
        logger.info(f"{node_name}: Checking 'conversational_tts' directly: {raw_conversational_tts_direct is not None}")
        if raw_conversational_tts_direct and str(raw_conversational_tts_direct).strip():
            conversational_tts = str(raw_conversational_tts_direct)
            logger.info(f"{node_name}: Using 'conversational_tts' directly: '{conversational_tts[:100]}...' ")
        else:
            output_content_from_state = state.get("output_content") or {}
            logger.info(f"{node_name}: Checking 'output_content' for TTS: {output_content_from_state.get('text_for_tts') is not None}")
            raw_tts_from_output_content = output_content_from_state.get("text_for_tts")
            if raw_tts_from_output_content and str(raw_tts_from_output_content).strip():
                conversational_tts = str(raw_tts_from_output_content)
                logger.info(f"{node_name}: Using 'text_for_tts' from 'output_content': '{conversational_tts[:100]}...' ")
            else:
                logger.info(f"{node_name}: 'text_for_tts' from 'output_content' is empty/None.")
    
    if not conversational_tts.strip():
        logger.info(f"{node_name}: No effective conversational TTS found. Will be empty string for tts_parts.")

    # Handle initial report content
    initial_report_content = state.get("initial_report_content")
    text_for_tts = ""
    if initial_report_content and isinstance(initial_report_content, dict) and initial_report_content.get("report_text"):
        text_for_tts = initial_report_content.get("report_text")
        logger.info(f"{node_name}: Using text from initial_report_content.")

    # Priority 2: Inactivity Prompt
    if not text_for_tts:
        output_content = state.get("output_content", {})
        if isinstance(output_content, dict) and output_content.get("text_for_tts"):
            text_for_tts = output_content["text_for_tts"]
            logger.info(f"{node_name}: Using text from output_content.text_for_tts (inactivity prompt).")


    # Priority 3: Pedagogy Output (e.g., task suggestion)
    if not text_for_tts:
        pedagogy_output = state.get("task_suggestion_llm_output", {})
        if isinstance(pedagogy_output, dict) and pedagogy_output.get("task_suggestion_tts"):
            text_for_tts = pedagogy_output["task_suggestion_tts"]
            logger.info(f"{node_name}: Using text from pedagogy_output.task_suggestion_tts.")


    # Fallback to the general conversational TTS content
    if not text_for_tts and conversational_tts:
        text_for_tts = conversational_tts
        logger.info(f"{node_name}: Using fallback text from conversational_tts.")

    tts_parts = []
    if text_for_tts:
        tts_parts.append(text_for_tts)
    
    consolidated_tts = " ".join(filter(None, tts_parts))
    logger.info(f"{node_name}: TTS parts collected: {tts_parts}")
    logger.info(f"{node_name}: Final combined TTS after join: '{consolidated_tts[:200]}...' (Length: {len(consolidated_tts)})")
    
    final_text_for_tts: str
    if not consolidated_tts.strip():
        logger.warning(f"{node_name}: No effective TTS strings found. Using default: 'I'm ready to help with your practice!'")
        final_text_for_tts = "I'm ready to help with your practice!"
    else:
        final_text_for_tts = consolidated_tts
        logger.info(f"{node_name}: Effective TTS string found: '{final_text_for_tts[:100]}...'" )

    client_response_data["final_text_for_tts"] = final_text_for_tts

    # --- Retrieve and Consolidate UI Actions ---
    accumulated_ui_actions: List[Dict[str, Any]] = []
    # current_context, task_stage, modelling_output_content, teaching_output are already available from TTS section

    # Prioritize modelling UI actions if in a modelling stage
    if "MODELLING" in task_stage:
        logger.info(f"{node_name}: Task stage '{task_stage}' indicates modelling. Prioritizing 'modelling_output_content' for UI actions.")
        modelling_actions = modelling_output_content.get("ui_actions")
        if modelling_actions is not None:
            if isinstance(modelling_actions, list) and modelling_actions:
                try:
                    accumulated_ui_actions.extend(modelling_actions)
                    logger.info(f"{node_name}: Extended ui_actions from 'modelling_output_content' (task stage priority). Count: {len(accumulated_ui_actions)}")
                except TypeError as e:
                    logger.error(f"{node_name}: TypeError extending ui_actions from 'modelling_output_content'. Error: {e}", exc_info=True)
            elif modelling_actions: # Not None, but not a non-empty list
                logger.warning(f"{node_name}: 'ui_actions' in 'modelling_output_content' (task stage priority) was not a non-empty list. Ignored.")

    # Fallback to teaching_output if modelling UI actions weren't used/found or not in modelling stage,
    # and accumulated_ui_actions is still empty.
    if not accumulated_ui_actions:
        logger.info(f"{node_name}: Checking 'teaching_output_content' for UI actions (task_stage: '{task_stage}'): {teaching_output.get('ui_actions') is not None}")
        teaching_actions = teaching_output.get("ui_actions")
        if teaching_actions is not None:
            if isinstance(teaching_actions, list) and teaching_actions:
                try:
                    accumulated_ui_actions.extend(teaching_actions)
                    logger.info(f"{node_name}: Extended ui_actions from 'teaching_output_content'. Count: {len(accumulated_ui_actions)}")
                except TypeError as e:
                    logger.error(f"{node_name}: TypeError extending ui_actions from 'teaching_output_content'. Error: {e}", exc_info=True)
            elif teaching_actions:
                logger.warning(f"{node_name}: 'ui_actions' in 'teaching_output_content' was not a non-empty list. Ignored.")

    if not accumulated_ui_actions:
        formatter_actions = state.get("ui_actions_for_formatter")
        if formatter_actions is not None:
            if isinstance(formatter_actions, list) and formatter_actions:
                try:
                    accumulated_ui_actions.extend(formatter_actions)
                    logger.info(f"{node_name}: Extended ui_actions from ui_actions_for_formatter. Count: {len(accumulated_ui_actions)}")
                except TypeError as e:
                    logger.error(f"{node_name}: TypeError extending ui_actions from ui_actions_for_formatter. Error: {e}", exc_info=True)
            elif formatter_actions:
                logger.warning(f"{node_name}: 'ui_actions_for_formatter' was not a non-empty list. Ignored.")

    if not accumulated_ui_actions:
        output_content_data = state.get("output_content")
        if isinstance(output_content_data, dict):
            oc_actions = output_content_data.get("ui_actions")
            if oc_actions is not None:
                if isinstance(oc_actions, list) and oc_actions:
                    try:
                        accumulated_ui_actions.extend(oc_actions)
                        logger.info(f"{node_name}: Extended ui_actions from output_content. Count: {len(accumulated_ui_actions)}")
                    except TypeError as e:
                        logger.error(f"{node_name}: TypeError extending ui_actions from output_content. Error: {e}", exc_info=True)
                elif oc_actions:
                    logger.warning(f"{node_name}: 'ui_actions' in output_content was not a non-empty list. Ignored.")
        elif output_content_data is not None:
            logger.warning(f"{node_name}: 'output_content' in state is not a dictionary. UI actions from it ignored.")

    final_ui_actions: List[Dict[str, Any]]
    if not isinstance(accumulated_ui_actions, list):
        logger.warning(f"{node_name}: 'accumulated_ui_actions' was not a list. Resetting to empty list.")
        final_ui_actions = []
    else:
        final_ui_actions = accumulated_ui_actions
    
    logger.info(f"{node_name}: Final 'ui_actions' for client response: {final_ui_actions} (Type: {type(final_ui_actions)})")
    client_response_data["final_ui_actions"] = final_ui_actions
    
    # --- Next Task Information ---
    next_task_details = state.get("next_task_details")
    if next_task_details is not None:
        client_response_data["final_next_task_info"] = next_task_details
        logger.info(f"{node_name}: Included 'final_next_task_info': {next_task_details}")
    else:
        client_response_data["final_next_task_info"] = None
        logger.info(f"{node_name}: 'next_task_details' not found. 'final_next_task_info' is None.")

    # --- Navigation Instruction ---
    nav_target = state.get("navigation_instruction_target") # Key from design document
    nav_data = state.get("data_for_target_page")       # Key from design document

    if nav_target:
        navigation_instruction = {"page_name": nav_target}
        if nav_data is not None:
            navigation_instruction["data"] = nav_data
        client_response_data["final_navigation_instruction"] = navigation_instruction
        logger.info(f"{node_name}: Included 'final_navigation_instruction': {navigation_instruction}")
    else:
        client_response_data["final_navigation_instruction"] = None
        logger.info(f"{node_name}: 'navigation_instruction_target' not found. 'final_navigation_instruction' is None.")

    # --- Raw System Outputs ---
    raw_teaching = state.get("teaching_output_content")
    if raw_teaching:
        client_response_data["raw_teaching_output"] = raw_teaching
        logger.info(f"{node_name}: Included 'raw_teaching_output'.")

    if modelling_output_content:
        client_response_data["raw_modelling_output"] = modelling_output_content
        logger.info(f"{node_name}: Included 'raw_modelling_output'.")
        if isinstance(modelling_output_content, dict):
            for key, value in modelling_output_content.items():
                if key not in ["text_for_tts", "ui_actions"]:
                    if key not in client_response_data:
                        client_response_data[key] = value
                        logger.info(f"{node_name}: Flattened '{key}' from modelling_output_content into response.")
                    else:
                        logger.warning(f"{node_name}: Key '{key}' from modelling_output_content already exists in response ('{client_response_data.get(key)}'). Not overwriting with '{value}'.")
        else:
            logger.warning(f"{node_name}: 'modelling_output_content' is not a dict, cannot flatten.")

    raw_feedback = state.get("feedback_output_content")
    if raw_feedback:
        client_response_data["raw_feedback_output"] = raw_feedback
        logger.info(f"{node_name}: Included 'raw_feedback_output'.")

    raw_pedagogy = state.get("task_suggestion_llm_output")
    if raw_pedagogy:
        client_response_data["raw_pedagogy_output"] = raw_pedagogy
        logger.info(f"{node_name}: Included 'raw_pedagogy_output'.")

    raw_initial_report = state.get("initial_report_content")
    if raw_initial_report:
        client_response_data["raw_initial_report_output"] = raw_initial_report
        logger.info(f"{node_name}: Included 'raw_initial_report_output'.")

    # Log summary of keys being returned
    logger.info(f"{node_name}: Final client response data keys: {list(client_response_data.keys())}")
    # Example of logging full content if small, or specific important parts:
    # logger.debug(f"{node_name}: Full client_response_data: {client_response_data}")

    logger.info(f"{node_name}: Value of 'raw_pedagogy_output' in returned dict: {client_response_data.get('raw_pedagogy_output')}")
    logger.info(f"{node_name}: Type of 'raw_pedagogy_output' in returned dict: {type(client_response_data.get('raw_pedagogy_output'))}")
    
    return client_response_data

