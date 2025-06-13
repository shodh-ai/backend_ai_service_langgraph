
import os
from dotenv import load_dotenv
load_dotenv() # Load .env variables at the very beginning


# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.concurrency import run_in_threadpool # Not explicitly used in new version, can be removed if not needed elsewhere
import logging
import uuid
import json
import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List # Keep for models if not fully typed
from pydantic import BaseModel, Field
from memory import mem0_memory # Import the shared instance

load_dotenv()
print(f"--- DEBUG: MEM0_API_KEY is set to: {os.getenv('MEM0_API_KEY')} ---")
from typing import Dict, Any, Optional, List

from graph_builder import build_graph
from state import AgentGraphState
from models import (
    InteractionRequest,
    InteractionResponse,
    InteractionRequestContext,
    ReactUIAction,
)

# Configure basic logging for the application
# This will set the root logger level and format, affecting all module loggers unless they are specifically configured otherwise.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# Load environment variables from .env file
load_dotenv()

# Configure logging
# Using uvicorn's logger for consistency if running with uvicorn
# BasicConfig can be used as a fallback or for non-uvicorn environments
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("uvicorn.error") # Standard for Uvicorn, captures its logs and app logs if configured

app = FastAPI(
    title="TOEFL Tutor AI Service",
    version="0.1.0",
    description="A LangGraph-based AI service for TOEFL tutoring."
)

# Add CORS middleware (from original app.py)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the graph when the application starts
toefl_tutor_graph = build_graph()

class UserRegistrationRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    name: str
    goal: str
    feeling: str
    confidence: str

@app.post("/user/register")
async def register_user(registration_data: UserRegistrationRequest):
    logger.info(f"Received registration data for user_id: {registration_data.user_id}")
    try:
        # The data from the form is already in a dictionary-like object.
        # We can convert it to a dict, excluding the user_id as that's the key for memory.
        profile_data = registration_data.dict(exclude={"user_id"})
        
        # Use the existing mem0_memory instance to update the student's profile
        mem0_memory.update_student_profile(
            user_id=registration_data.user_id,
            profile_data=profile_data
        )
        
        logger.info(f"Successfully stored profile for user_id: {registration_data.user_id}")
        return {"message": "User profile stored successfully."}
    except Exception as e:
        logger.error(f"Failed to store user profile for {registration_data.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store user profile.")



async def stream_graph_responses_sse(request_data: InteractionRequest):
    """
    Asynchronously streams graph execution events as Server-Sent Events (SSE).
    Handles 'streaming_text_chunk' for live text updates and the final
    consolidated response from the output formatter node.
    """
    default_user_id = "default_user_for_streaming_test"
    default_session_id = str(uuid.uuid4())

    user_id = default_user_id
    if request_data.current_context and request_data.current_context.user_id:
        user_id = request_data.current_context.user_id

    session_id = request_data.session_id or default_session_id
    context = request_data.current_context or InteractionRequestContext(user_id=user_id)
    chat_history = request_data.chat_history
    transcript = request_data.transcript
    full_submitted_transcript = None

    if context.task_stage == "speaking_task_submitted":
        full_submitted_transcript = transcript

    # Prepare initial state similar to the non-streaming endpoint
    initial_graph_state: AgentGraphState = {
        "user_id": user_id,
        "user_token": request_data.usertoken,
        "session_id": session_id,
        "transcript": transcript,
        "full_submitted_transcript": full_submitted_transcript,
        "current_context": context,
        "chat_history": chat_history,
        "question_stage": context.question_stage,
        # Initialize other fields as in the non-streaming endpoint
        "student_memory_context": None,
        "next_task_details": None,
        "diagnosis_result": None,
        "output_content": None, # Will be populated by output_formatter
        "feedback_content": None,
        "estimated_overall_english_comfort_level": context.english_comfort_level,
        "initial_impression": context.teacher_initial_impression,
        "fluency": context.fluency,
        "grammar": context.grammar,
        "vocabulary": context.vocabulary,
        "question_one_answer": context.question_one_answer,
        "question_two_answer": context.question_two_answer,
        "question_three_answer": context.question_three_answer,
        "example_prompt_text": context.example_prompt_text,
        "modelling_output_content": None,
        "teaching_output_content": None,
        "task_suggestion_llm_output": None,
        "inactivity_prompt_response": None,
        "motivational_support_response": None,
        "tech_support_response": None,
        "navigation_instruction_target": None,
        "data_for_target_page": None,
        "conversational_tts": None, # This will be superseded by streaming_text_chunk
        "cowriting_output_content": None, 
        "scaffolding_output_content": None,
        "session_summary_text": None,
        "progress_report_text": None,
        "student_model_summary": None,
        "system_prompt_config": None,
        "llm_json_validation_map": None,
        "error_count": 0,
        "last_error_message": None,
        "current_node_name": None
    }

    config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
    logger.info(f"Streaming endpoint: Initializing graph stream for session {session_id}, user {user_id}")

    try:
        # Use astream_events_v2 to get detailed events
        async for event in toefl_tutor_graph.astream_events(initial_graph_state, config=config, stream_mode="values", output_keys=["output_content"]):
            event_name = event.get("event")
            node_name = event.get("name") # Name of the node that produced the event
            data = event.get("data", {})
            tags = event.get("tags", [])

            # logger.debug(f"SSE Stream Event: {event_name}, Node: {node_name}, Data: {data}, Tags: {tags}")

            if event_name == "on_chain_stream" and data:
                chunk_content = data.get("chunk") # LangGraph's default key for streaming output from .stream()
                if isinstance(chunk_content, dict):
                    # Check for intermediate streaming text chunks. This is the correct place for this.
                    streaming_text = chunk_content.get("streaming_text_chunk")
                    if streaming_text:
                        logger.debug(f"SSE Stream: Yielding intermediate 'streaming_text_chunk' from node '{node_name}': {streaming_text[:100]}...")
                        yield f"event: streaming_text_chunk\ndata: {json.dumps({'streaming_text_chunk': streaming_text})}\n\n"
                        await asyncio.sleep(0.01)

            elif event_name == "on_chain_end" and node_name == "format_final_output":
                # This event signifies the end of the graph's main processing.
                # The node's return value is in the 'output' key of the 'on_chain_end' event data.
                final_output_package = data.get('output', {})
                
                if isinstance(final_output_package, dict):
                    # Extract the TTS text and UI actions directly from the node's output
                    tts_text = final_output_package.get("final_text_for_tts")
                    ui_actions = final_output_package.get("final_ui_actions")

                    if tts_text:
                        logger.info(f"SSE Stream: Yielding final TTS text chunk.")
                        yield f"event: streaming_text_chunk\ndata: {json.dumps({'streaming_text_chunk': tts_text})}\n\n"
                    
                    if ui_actions:
                        logger.info(f"SSE Stream: Yielding final UI actions.")
                        yield f"event: final_ui_actions\ndata: {json.dumps({'ui_actions': ui_actions})}\n\n"
                        await asyncio.sleep(0.01)

            elif event_name == "on_chain_end" and node_name == "error_generator_node":
                # 'data' directly contains the output of the node when stream_mode="values".
                error_output = data 
                if isinstance(error_output, dict) and error_output.get("output_content"):
                    error_response_data = error_output.get("output_content")
                    logger.error(f"SSE Stream: Yielding error_response: {json.dumps(error_response_data)}")
                    yield f"event: error_response\ndata: {json.dumps(error_response_data)}\n\n"
    finally:
        logger.info(f"SSE Stream: Closing stream for session {session_id}")
        yield f"event: stream_end\ndata: {{'message': 'Stream ended'}}\n\n"

@app.post("/process_interaction_streaming")
async def process_interaction_streaming_route(request_data: InteractionRequest):
    logger.info(f"Received request for /process_interaction_streaming: user_id={request_data.current_context.user_id if request_data.current_context else 'N/A'}, session_id={request_data.session_id}")
    return StreamingResponse(stream_graph_responses_sse(request_data), media_type="text/event-stream")


@app.post("/process_interaction", response_model=InteractionResponse)
async def process_interaction_route(request_data: InteractionRequest):
    default_user_id = "default_user_for_testing"
    default_session_id = str(uuid.uuid4())

    user_id = default_user_id
    if request_data.current_context and request_data.current_context.user_id:
        user_id = request_data.current_context.user_id

    session_id = request_data.session_id or default_session_id
    context = request_data.current_context or InteractionRequestContext(user_id=user_id)
    chat_history = request_data.chat_history
    transcript = request_data.transcript
    full_submitted_transcript = None

    if context.task_stage == "speaking_task_submitted":
        full_submitted_transcript = transcript

    initial_graph_state: AgentGraphState = {
        "user_id": user_id,
        "user_token": request_data.usertoken,
        "session_id": session_id,
        "transcript": transcript,
        "full_submitted_transcript": full_submitted_transcript,
        "current_context": context,
        "chat_history": chat_history,
        "question_stage": context.question_stage,
        "student_memory_context": None,
        "next_task_details": None,
        "diagnosis_result": None,
        "output_content": None,
        "feedback_content": None,

        # --- Fields from merged branches ---
        # Populate all fields to ensure nodes from both branches work.

        # 'feedback-system' fields, mapped from available context
        "estimated_overall_english_comfort_level": context.english_comfort_level,
        "initial_impression": context.teacher_initial_impression,
        "fluency": context.fluency,
        "grammar": context.grammar,
        "vocabulary": context.vocabulary,
        "question_one_answer": context.question_one_answer,
        "question_two_answer": context.question_two_answer,
        "question_three_answer": context.question_three_answer,

        # 'teaching&modelling' fields
        "example_prompt_text": context.example_prompt_text,
        "student_goal_context": context.student_goal_context,
        "student_confidence_context": context.student_confidence_context,
        "teacher_initial_impression": context.teacher_initial_impression,
        "student_struggle_context": context.student_struggle_context,
        "english_comfort_level": context.english_comfort_level,
    }

    try:
        return StreamingResponse(stream_graph_responses_sse(request_data), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {e}", exc_info=True)
        # It's important to raise HTTPException for FastAPI to handle it correctly
        raise HTTPException(status_code=500, detail=f"Error processing streaming request: {str(e)}")

# Renamed original endpoint for non-streaming for clarity and to avoid conflict
@app.post("/process_interaction_non_streaming", response_model=InteractionResponse)
async def process_interaction_non_streaming_route(request_data: InteractionRequest):
    logger.info(f"Non-streaming request received for user '{request_data.current_context.user_id}' session '{request_data.session_id}'")
    try:
        initial_graph_state = AgentGraphState(
            user_id=request_data.current_context.user_id,
            session_id=request_data.session_id,
            transcript=request_data.transcript,
            current_context=request_data.current_context,
            chat_history=request_data.chat_history,
            user_token=request_data.usertoken,
            full_submitted_transcript=request_data.transcript if request_data.current_context.task_stage == "speaking_task_submitted" else None,
            question_stage=request_data.current_context.question_stage,
            student_memory_context=None,
            task_stage=request_data.current_context.task_stage,
            next_task_details=None,
            diagnosis_result=None,
            error_details=None,
            document_query_result=None,
            rag_query_result=None,
            feedback_plan=None,
            feedback_output=None,
            feedback_content=None,
            scaffolding_analysis=None,
            scaffolding_retrieval_result=None,
            scaffolding_plan=None,
            scaffolding_output=None,
            teaching_module_state=None,
            p1_curriculum_navigator_output=None,
            conversation_response=None,
            output_content=None,
        )

        # Prepare config for LangGraph invocation, crucial for Mem0 checkpointer
        config = {
            "configurable": {
                "thread_id": request_data.session_id, # Using session_id as thread_id
                "user_id": request_data.current_context.user_id # Optional: if user_id is also useful in config
            }
        }
        logger.info(f"Non-streaming endpoint: Invoking graph for session {request_data.session_id}, user {request_data.current_context.user_id}")

        # Invoke the graph
        # The input to ainvoke should be the initial_graph_state or a subset of it
        # that matches the graph's expected input schema.
        # For AgentGraphState, we pass the whole state dict as the primary input.
        final_state = await toefl_tutor_graph.ainvoke(
            input=initial_graph_state,  # Pass the fully prepared initial state
            config=config
        )

        # --- BEGIN Detailed final_state logging ---
        logger.warning(f"APP.PY: Received final_state type: {type(final_state)}")
        if isinstance(final_state, dict):
            logger.warning(f"APP.PY: final_state keys: {list(final_state.keys())}")
            final_tts_content_from_log = final_state.get('final_text_for_tts') # Use a different var name to avoid confusion with response_text
            logger.warning(f"APP.PY: final_state content for 'final_text_for_tts': '{str(final_tts_content_from_log)[:200]}...' (Type: {type(final_tts_content_from_log)})")
            # Log a few other potentially relevant keys from output_formatter_node
            logger.warning(f"APP.PY: final_state content for 'final_ui_actions': {final_state.get('final_ui_actions')}")
            logger.warning(f"APP.PY: final_state content for 'raw_modelling_output': {'present' if 'raw_modelling_output' in final_state else 'missing'}")
        elif hasattr(final_state, '__dict__'):
            logger.warning(f"APP.PY: final_state is an object. Attributes: {list(final_state.__dict__.keys())}")
            final_tts_content_from_log = getattr(final_state, 'final_text_for_tts', 'AttributeNotPresent')
            logger.warning(f"APP.PY: final_state attribute 'final_text_for_tts': '{str(final_tts_content_from_log)[:200]}...' (Type: {type(final_tts_content_from_log)})")
        else:
            logger.warning(f"APP.PY: final_state is not a dict and has no __dict__. Dir: {dir(final_state)}")
            logger.warning(f"APP.PY: final_state raw content: {str(final_state)[:500]}...") # Log a snippet if unknown type
        # --- END Detailed final_state logging ---

        # Extract final outputs from the graph state based on the new structure
        response_text = final_state.get("final_text_for_tts")
        if not response_text:
            response_text = "I'm ready for your next instruction. Please let me know how I can help!"
            logger.warning("final_text_for_tts not found or empty in final_state. Using default message.")

        # final_ui_actions from state are expected to be List[Dict[str, Any]]
        current_ui_action_dicts: List[Dict[str, Any]] = final_state.get("final_ui_actions") or []

        next_task_info = final_state.get("final_next_task_info")
        navigation_instruction = final_state.get("final_navigation_instruction")

        # Logic to add a default UI action for next_task if not already present
        if next_task_info:
            has_task_button = any(
                action.get("action_type") == "DISPLAY_NEXT_TASK_BUTTON"
                for action in current_ui_action_dicts
            )
            # Also consider if a navigation action might implicitly handle the next task display
            navigates_to_task = False
            if navigation_instruction and navigation_instruction.get("data"):
                # This is a heuristic; actual task pages might vary
                if "task_id" in navigation_instruction.get("data", {}) or \
                   next_task_info.get("prompt_id") == navigation_instruction.get("data", {}).get("prompt_id") :
                   navigates_to_task = True
            
            if not has_task_button and not navigates_to_task:
                task_title = next_task_info.get("title", "New Task Available")
                task_desc = next_task_info.get("description", "Please check your tasks.")
                # Using SHOW_ALERT as a fallback, consider if a more specific action is better
                current_ui_action_dicts.append({
                    "action_type": "SHOW_ALERT",
                    "parameters": {"message": f"Next Task: {task_title}\n{task_desc}"}
                })
                logger.info(f"Added SHOW_ALERT UI action for next_task_info: {task_title}")

        # Convert ui_action dictionaries to ReactUIAction Pydantic models
        ui_actions_list: Optional[List[ReactUIAction]] = None
        if current_ui_action_dicts:
            ui_actions_list = []
            for action_dict in current_ui_action_dicts:
                action_type_str = action_dict.get("action_type", "") # Expect 'action_type' directly
                
                target_element_id = action_dict.get("target_element_id")
                # Handle potential camelCase from frontend or other systems if necessary, though backend should be consistent
                if target_element_id is None and "targetElementId" in action_dict:
                    target_element_id = action_dict["targetElementId"]
                    logger.debug("Used 'targetElementId' for target_element_id")

                parameters = action_dict.get("parameters", {})

                try:
                    ui_actions_list.append(
                        ReactUIAction(
                            action_type=action_type_str,
                            target_element_id=target_element_id,
                            parameters=parameters,
                        )
                    )
                except Exception as pydantic_exc:
                    logger.error(f"Error creating ReactUIAction for dict {action_dict}: {pydantic_exc}", exc_info=True)
                    # Optionally, skip this action or add a default error action

        # Construct the final InteractionResponse
        # Construct the dictionary for InteractionResponse instantiation.
        # This includes all fields from final_state, allowing raw outputs and
        # flattened fields to be passed as 'extra' fields due to `extra = Extra.allow`
        # in the Pydantic model. The explicitly mapped fields (`response`, `ui_actions`, etc.)
        # will override any same-named keys from final_state.
        model_init_kwargs = {
            **final_state,  # Spread all of final_state
            "response": response_text,  # Set/override with processed TTS text
            "ui_actions": ui_actions_list,  # Set/override with processed UIAction objects
            "next_task_info": next_task_info,  # Set/override
            "navigation_instruction": navigation_instruction,  # Set/override
        }

        response = InteractionResponse(**model_init_kwargs)

        # Optional: Log the keys that will be sent for verification
        # logger.debug(f"InteractionResponse being sent with keys: {list(response.model_dump().keys())}")


        return InteractionResponse(response=response_text, ui_actions=ui_actions_list, session_id=request_data.session_id)

    except Exception as e:

        logger.error(f"Exception in /process_interaction: {e}", exc_info=True)
        # The following import and print_exc are for more verbose console output if needed, 
        # but logger.error with exc_info=True should capture it well.
        # import traceback
        # traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error during AI processing: {str(e)}"
        )



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    # Use PORT from env, default to 8000 (common for dev, original used 5005)
    port = int(os.getenv("PORT", "8001")) 
    logger.info(f"Starting Uvicorn server on host 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
