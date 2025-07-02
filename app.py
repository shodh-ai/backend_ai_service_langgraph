import os
import asyncio
from dotenv import load_dotenv

# Load environment variables only once at the beginning
load_dotenv()
if os.getenv("MEM0_API_KEY"):
    print(f"--- DEBUG: MEM0_API_KEY is set. ---")
else:
    print("--- WARNING: MEM0_API_KEY is NOT set. Mem0 functionalities will fail. ---")

import logging
# Configure basic logging early, this might be reconfigured by uvicorn but good for initial script execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("uvicorn.error") # Ensure logger is defined before use in endpoints

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field # For UserRegistrationRequest
from fastapi import UploadFile, File
from deepgram import DeepgramClient
from deepgram import PrerecordedOptions

from memory import initialize_memory
import memory
 
from graph_builder import build_graph
from state import AgentGraphState
import uuid

# Define the directory for uploads
UPLOAD_DIR = "uploads"

# +++ DEFINE THE NEW REQUEST MODEL +++
# This matches the payload that livekit-service is sending.
class InvokeTaskRequest(BaseModel):
    task_name: str
    json_payload: str
from contextlib import asynccontextmanager
from models import (
    InteractionRequest,
    InteractionResponse,
    InteractionRequestContext,
    ReactUIAction,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

toefl_tutor_graph: Optional[Any] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes the global memory instance and the graph with its checkpointer.
    """
    global toefl_tutor_graph
    logger.info("--- Application startup: Initializing memory and graph ---")
    
    initialize_memory()
    logger.info("--- Application startup: StudentProfileMemory initialized ---")

    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        toefl_tutor_graph = build_graph(checkpointer)
        logger.info("--- Application startup: Main graph compiled with SQLite checkpointer ---")
        
        yield
    
    logger.info("--- Application shutdown: Resources cleaned up ---")


app = FastAPI(
    title="ShodhAI Langgraph",
    version="0.1.0",
    description="A LangGraph-based AI service for academic tutoring.",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The graph is now initialized within the lifespan manager
# toefl_tutor_graph = build_graph()

# --- Deepgram Transcription Proxy Endpoint ---
@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    logger.info("Received request for /transcribe_audio")
    try:
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not deepgram_api_key:
            logger.error("DEEPGRAM_API_KEY not set in environment.")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing Deepgram API key.")

        # Initialize Deepgram client
        deepgram = DeepgramClient(deepgram_api_key)

        audio_data = await file.read()
        source = {'buffer': audio_data, 'mimetype': file.content_type}

        # Set transcription options
        options = PrerecordedOptions(
            model='nova-2',
            smart_format=True,
            language='en-US'
        )

        # Call Deepgram API
        logger.info("Sending audio to Deepgram for transcription...")
        response = await deepgram.listen.asyncprerecorded.v("1").transcribe_file(source, options)
        
        # Extract transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        logger.info(f"Successfully transcribed audio. Transcript: {transcript[:100]}...")
        
        return {"transcript": transcript}

    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")


# --- User Registration Endpoint (from origin/subgraphs) ---
class UserRegistrationRequest(BaseModel):
    user_id: str = Field(..., description="The unique identifier for the user.")
    name: str
    goal: str
    feeling: str
    confidence: str

async def create_initial_state(request_data: InvokeTaskRequest) -> AgentGraphState:
    """
    Build the state for *this turn only* from the incoming request.
    No attempt is made to load or merge prior state – LangGraph's
    checkpointer handles persistence and merging automatically.
    """
    # 1. Parse the incoming JSON payload safely
    try:
        payload = json.loads(request_data.json_payload)
    except json.JSONDecodeError:
        logger.error("Failed to decode json_payload. Using empty payload.")
        payload = {}

    # 2. Core identifiers
    user_id = payload.get("user_id", "unknown_user")
    session_id = payload.get("session_id", str(uuid.uuid4()))

    # 3. Current turn context (can be empty or partial)
    incoming_context = payload.get("current_context", {})
    # Ensure at minimum the user_id is present for downstream logic
    incoming_context.setdefault("user_id", user_id)

    # 4. Determine task_name for routing. Prefer the explicit `task_name` from the request object,
    #    but fall back to any task_stage present in the incoming_context.
    task_name = request_data.task_name or incoming_context.get("task_stage")

    logger.info(f"[create_initial_state] Building turn state for task: '{task_name}' | session_id: '{session_id}'")

    # 5. Construct the minimal state dict. Fields not set here will be
    #    populated by the graph or restored from the checkpoint after merge.
    initial_state: AgentGraphState = {
        "user_id": user_id,
        "session_id": session_id,
        "task_name": task_name,

        # Turn-specific inputs
        "transcript": payload.get("transcript"),
        "chat_history": payload.get("chat_history", []),
        "incoming_context": incoming_context, # Use the new dedicated key

        # Place-holders / defaults for the remainder of the AgentGraphState keys.
        # They will either be set by nodes or restored by the checkpointer merge.
        "current_context": {}, # Will be populated from checkpoint
        "rag_document_data": [],
        "intermediate_modelling_payload": None,
        "intermediate_teaching_payload": None,
        "intermediate_scaffolding_payload": None,
        "intermediate_feedback_payload": None,
        "intermediate_cowriting_payload": None,
        "intermediate_pedagogy_payload": None,
    }

    return initial_state

@app.post("/invoke_task_streaming")
async def invoke_task_streaming_route(request_data: InvokeTaskRequest):
    """
    Receives a generic task, unpacks the payload, and starts the graph stream.
    This is the new, preferred entry point.
    """
    try:
        initial_graph_state = await create_initial_state(request_data)

        # Create the config for the graph invocation, which is essential for memory
        config = {"configurable": {"thread_id": initial_graph_state.get("session_id")}, "recursion_limit": 50}

        # Call the SSE streamer with the prepared state and config
        return StreamingResponse(
            stream_graph_responses_sse(initial_graph_state, config),
            media_type="text/event-stream"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode json_payload: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in payload: {e}")
    except Exception as e:
        logger.error(f"Error in invoke_task_streaming: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/user/register")
async def register_user(registration_data: UserRegistrationRequest):
    logger.info(f"Received registration data for user_id: {registration_data.user_id}")
    try:
        profile_data = registration_data.model_dump(exclude={"user_id"})
        
        memory.memory_stub.update_student_profile(
            user_id=registration_data.user_id,
            profile_data=profile_data
        )
        
        logger.info(f"Successfully stored profile for user_id: {registration_data.user_id}")
        return {"message": "User profile stored successfully."}
    except Exception as e:
        logger.error(f"Failed to store user profile for {registration_data.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store user profile.")

# --- Streaming Endpoint --- 
# In app.py

# FINAL, PERFECTED app.py

async def stream_graph_responses_sse(initial_graph_state: AgentGraphState, config: dict):
    """
    This final version listens for the output of a finalizing node and
    streams it back as a single, comprehensive "final_response" event.
    """
    try:
        yield f"event: stream_start\ndata: {json.dumps({'message': 'Stream started'})}\n\n"
        
        FINALIZING_NODES = [
            "conversation_handler", "handle_welcome", "acknowledge_interrupt",
            "modelling_output_formatter", "teaching_output_formatter",
            "scaffolding_output_formatter", "feedback_output_formatter",
            "cowriting_output_formatter", "pedagogy_output_formatter",
        ]

        async for event in toefl_tutor_graph.astream_events(initial_graph_state, config=config, stream_mode="values"):
            event_name = event.get("event")
            node_name = event.get("name")
            
            if event_name == "on_chain_end" and node_name in FINALIZING_NODES:
                logger.info(f"SSE Streamer: Captured final output from node '{node_name}'.")
                output_data = event.get("data", {}).get("output", {})
                
                if not output_data:
                    logger.warning(f"Node '{node_name}' finished but produced no output data.")
                    continue

                # --- THIS IS THE FIX ---
                # We package everything into ONE event.
                final_response_payload = {
                    "final_text_for_tts": output_data.get("final_text_for_tts"),
                    "final_ui_actions": output_data.get("final_ui_actions", [])
                }

                logger.info("SSE Streamer: Yielding single 'final_response' event.")
                yield f"event: final_response\ndata: {json.dumps(final_response_payload)}\n\n"

    except Exception as e:
        logger.error(f"Error streaming graph responses: {e}", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    finally:
        logger.info("SSE stream finished.")
        yield f"event: stream_end\ndata: {json.dumps({'message': 'Stream complete'})}\n\n"

@app.post("/process_interaction_streaming")
async def process_interaction_streaming_route(request_data: InteractionRequest):
    """
    Process an interaction and stream back responses as SSE.
    This old endpoint can now be simplified to be a wrapper around the new logic.
    """
    logger.warning("DEPRECATED: /process_interaction_streaming called. Please migrate to /invoke_task_streaming.")
    
    # Convert the old request model into the new format.
    # This acts as an "adapter" so you don't break old clients.
    task_name = "unknown_legacy_task" # You might parse this from the request if possible
    
    # Create the initial state just like the old endpoint did.
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

    # Extract next_task_details from request payload if present
    next_task_details = None
    
    # Method 1: Direct attribute access
    if hasattr(request_data, 'next_task_details'):
        if request_data.next_task_details is not None:
            next_task_details = request_data.next_task_details
    
    # Method 2: Dictionary extraction
    request_dict = request_data.model_dump()
    if 'next_task_details' in request_dict:
        dict_task_details = request_dict.get('next_task_details')
        if next_task_details is None and dict_task_details is not None:
            next_task_details = dict_task_details
    
    initial_graph_state: AgentGraphState = {
        "user_id": user_id,
        "user_token": getattr(request_data, "user_token", None),
        "session_id": session_id,
        "transcript": transcript,
        "full_submitted_transcript": full_submitted_transcript,
        "current_context": context.model_dump() if context else {},
        "chat_history": [msg.model_dump() for msg in chat_history] if chat_history else [],
        "task_name": task_name,
        "question_stage": getattr(context, "question_stage", None),
        "student_memory_context": None,
        "next_task_details": next_task_details,
        "diagnosis_result": None,
        "output_content": None, 
        "feedback_content": None,
        "estimated_overall_english_comfort_level": getattr(context, "english_comfort_level", None),
        "initial_impression": getattr(context, "teacher_initial_impression", None),
        "fluency": getattr(context, "fluency", None),
        "grammar": getattr(context, "grammar", None),
        "vocabulary": getattr(context, "vocabulary", None),
        "goal": getattr(context, "goal", None),
        "feeling": getattr(context, "feeling", None),
        "confidence": getattr(context, "confidence", None),
        "example_prompt_text": getattr(context, "example_prompt_text", None),
        "student_goal_context": getattr(context, "student_goal_context", None),
        "student_confidence_context": getattr(context, "student_confidence_context", None),
        "teacher_initial_impression": getattr(context, "teacher_initial_impression", None),
        "student_struggle_context": getattr(context, "student_struggle_context", None),
        "english_comfort_level": getattr(context, "english_comfort_level", None),
        "modelling_output_content": None,
        "teaching_output_content": None,
        "task_suggestion_llm_output": None,
        "inactivity_prompt_response": None,
        "motivational_support_response": None,
        "tech_support_response": None,
        "navigation_instruction_target": None,
        "data_for_target_page": None,
        "conversational_tts": None,
        "cowriting_output_content": None, 
        "scaffolding_output_content": None,
        "session_summary_text": None,
        "progress_report_text": None,
        "student_model_summary": None,
        "system_prompt_config": None,
        "llm_json_validation_map": None,
        "error_count": 0,
        "last_error_message": None,
        "current_node_name": None,
    }
    
    # Prepare config for LangGraph invocation
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 50,
    }
    
    # Call the *same* reusable SSE function.
    return StreamingResponse(
        stream_graph_responses_sse(initial_graph_state, config=config), 
        media_type="text/event-stream"
    )

@app.post("/invoke_task", response_model=Dict[str, Any])
async def invoke_task_route(request_data: InvokeTaskRequest):
    """
    Handles non-streaming requests for fast, simple nodes.
    """
    logger.info(f"Received non-streaming task '{request_data.task_name}'.")
    try:
        initial_state = await create_initial_state(request_data)
        config = {"configurable": {"thread_id": initial_state["session_id"]}, "recursion_limit": 50}
        
        # Use ainvoke for a single, final result.
        final_state = await toefl_tutor_graph.ainvoke(initial_state, config=config)
        
        # Return the final output keys directly.
        return {
            "final_text_for_tts": final_state.get("final_text_for_tts"),
            "final_ui_actions": final_state.get("final_ui_actions", [])
        }
    except Exception as e:
        logger.error(f"Error in non-streaming invoke_task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process_interaction", response_model=InteractionResponse)
async def process_interaction_route(request_data: InteractionRequest):
    # Debugging: Log the entire request payload to see what we're getting
    logger.info(f"Received request payload: {json.dumps(request_data.model_dump(), default=str)[:500]}...")

    # --- ADDED FOR TOKEN DEBUGGING ---
    user_token = request_data.user_token
    logger.info(f"--- TOKEN RECEIVED IN /process_interaction ---")
    logger.info(f"USER_TOKEN: {user_token}")
    logger.info(f"----------------------------------------------")
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

    # Extract next_task_details from request payload if present
    next_task_details = None
    
    # Method 1: Direct attribute access
    if hasattr(request_data, 'next_task_details'):
        logger.info(f"next_task_details attribute exists: {hasattr(request_data, 'next_task_details')}")
        if request_data.next_task_details is not None:
            next_task_details = request_data.next_task_details
            logger.info(f"Method 1: Found next_task_details in request: {next_task_details}")
    
    # Method 2: Dictionary extraction
    request_dict = request_data.model_dump()
    logger.info(f"Request dict keys: {list(request_dict.keys())}")
    if 'next_task_details' in request_dict:
        dict_task_details = request_dict.get('next_task_details')
        logger.info(f"Method 2: Extracted next_task_details from dict: {dict_task_details}")
        if next_task_details is None and dict_task_details is not None:
            next_task_details = dict_task_details
    
    # Override from top-level fields for testing
    if next_task_details is None or not isinstance(next_task_details, dict) or 'page_target' not in next_task_details:
        # Create a valid next_task_details for testing
        next_task_details = {
            "title": "Reading Comprehension: Academic Passages",
            "type": "practice",
            "page_target": "reading_exercise",
            "prompt_id": "read_comprehension_001"
        }
        logger.info(f"Created test next_task_details: {next_task_details}")
    
    initial_graph_state: AgentGraphState = {
        "user_id": user_id,
        "user_token": request_data.user_token,
        "session_id": session_id,
        "transcript": transcript,
        "full_submitted_transcript": full_submitted_transcript,
        "current_context": context,
        "chat_history": chat_history,
        "question_stage": context.question_stage,
        "student_memory_context": None,
        "task_stage": request_data.current_context.task_stage,
        "next_task_details": next_task_details,
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
        "goal": context.goal,
        "feeling": context.feeling,
        "confidence": context.confidence,

        # 'teaching&modelling' fields
        "example_prompt_text": context.example_prompt_text,
        "student_goal_context": context.student_goal_context,
        "student_confidence_context": context.student_confidence_context,
        "teacher_initial_impression": context.teacher_initial_impression,
        "student_struggle_context": context.student_struggle_context,
        "english_comfort_level": context.english_comfort_level,
    }

    try:
        # Prepare config for LangGraph invocation
        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            },
            "recursion_limit": 50
        }
        
        # Use regular ainvoke for non-streaming response
        final_state = await toefl_tutor_graph.ainvoke(initial_graph_state, config=config)
        
        # Extract response components
        # Log all state keys for debugging
        logger.info(f"Final state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
        logger.info(f"Final next_task_details: {final_state.get('next_task_details')}")

        # Extract final TTS text from the output
        # First check if final_text_for_tts exists
        text_for_tts = None
        raw_pedagogy_output = None
        raw_initial_report = None
        raw_inactivity_output = None
        
        # First try to get from final_text_for_tts (preferred)
        if "final_text_for_tts" in final_state:
            text_for_tts = final_state["final_text_for_tts"]
        # Then try to get from task_suggestion_llm_output if present
        elif "task_suggestion_llm_output" in final_state and final_state["task_suggestion_llm_output"]:
            pedagogy_output = final_state["task_suggestion_llm_output"]
            if isinstance(pedagogy_output, dict) and "task_suggestion_tts" in pedagogy_output:
                text_for_tts = pedagogy_output["task_suggestion_tts"]
                raw_pedagogy_output = pedagogy_output
        # Then check for initial_report_content
        elif "initial_report_content" in final_state:
            report_content = final_state["initial_report_content"]
            if isinstance(report_content, dict) and "report_text" in report_content:
                text_for_tts = report_content["report_text"]
                raw_initial_report = report_content
        # Finally check for inactivity output
        elif "output_content" in final_state and isinstance(final_state.get("output_content", {}), dict):
            output_content = final_state["output_content"]
            if "text_for_tts" in output_content:
                text_for_tts = output_content["text_for_tts"]
                raw_inactivity_output = output_content
        
        if text_for_tts is None:
            logger.warning(f"Non-streaming response has no text_for_tts available in keys: {list(final_state.keys())}")
            text_for_tts = "I don't have a response at this time."

        # Create response with proper field names according to InteractionResponse model
        response = InteractionResponse(
            response=text_for_tts,  # This is the main TTS field in InteractionResponse
            ui_actions=final_state.get("final_ui_actions", []),
            next_task_info=final_state.get("final_next_task_info"),
            navigation_instruction=final_state.get("final_navigation_instruction"),
            raw_initial_report_output=raw_initial_report,
            # Add the following as extra fields (allowed by model_config={"extra": "allow"})
            raw_pedagogy_output=raw_pedagogy_output,
            raw_inactivity_output=raw_inactivity_output,
            raw_teaching_output=final_state.get("raw_teaching_output"),
            raw_modelling_output=final_state.get("raw_modelling_output"),
            raw_feedback_output=final_state.get("raw_feedback_output")
        )
        
        return response
    except Exception as e:
        logger.error(f"Error in processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Renamed original endpoint for non-streaming for clarity and to avoid conflict
@app.post("/process_interaction_non_streaming", response_model=InteractionResponse)
async def process_interaction_non_streaming_route(request_data: InteractionRequest):
    logger.info(f"Non-streaming request received for user '{request_data.current_context.user_id}' session '{request_data.session_id}'")
    try:
        # Initialize state tracking variables
        middleware_state = {}
        
        initial_graph_state = AgentGraphState(
            user_id=request_data.current_context.user_id,
            session_id=request_data.session_id,
            transcript=request_data.transcript,
            current_context=request_data.current_context,
            chat_history=request_data.chat_history,
            user_token=request_data.user_token,
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
            },
            "recursion_limit": 50
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


        return response

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


    # Ensure the upload directory exists
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    
    # Run the FastAPI app with uvicorn
    port = int(os.getenv("PORT", "8080")) 
    logger.info(f"Starting Uvicorn server on host 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
