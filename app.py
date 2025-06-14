from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import socketio
import asyncio
from graph_builder import build_graph
from state import AgentGraphState
from models import (
    InteractionRequest,
    InteractionResponse,
    InteractionRequestContext,
    ReactUIAction,
)
import io
import base64
from gtts import gTTS

# Configure basic logging for the application
# This will set the root logger level and format, affecting all module loggers unless they are specifically configured otherwise.
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

load_dotenv()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="TOEFL Tutor AI Backend", version="0.1.0")

# Create a Socket.IO server instance
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow connections from Next.js frontend
    logger=True,  # Enable logging
    engineio_logger=True  # Enable Engine.IO logging
)

# Create a Socket.IO ASGI application
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='socket.io'  # This matches the client's default path
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

toefl_tutor_graph = build_graph()


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

    initial_graph_state = AgentGraphState(
        user_id=user_id,
        user_token=request_data.usertoken,
        session_id=session_id,
        transcript=transcript,
        full_submitted_transcript=full_submitted_transcript,
        current_context=context,
        chat_history=chat_history,
        question_stage=context.question_stage,
        student_memory_context=None,
        next_task_details=None,
        diagnosis_result=None,
        output_content=None,
        feedback_content=None,
    )

    try:
        config = {"configurable": {"thread_id": session_id}}
        final_state = await toefl_tutor_graph.ainvoke(
            initial_graph_state, config=config
        )

        output_content: Optional[Dict[str, Any]] = final_state.get("output_content")
        if output_content is None:
            output_content = final_state.get("feedback_content", {})

        response_text = ""
        if output_content:
            response_text = output_content.get(
                "response",
                output_content.get("text_for_tts", output_content.get("text", "")),
            )

        if not response_text:
            response_text = (
                "No response text was generated. Please check the system logs."
            )

        ui_actions = None
        if output_content:
            ui_actions = output_content.get("ui_actions") or output_content.get(
                "dom_actions"
            )
            if ui_actions:
                for action in ui_actions:
                    if "action_type" in action:
                        action["action_type_str"] = action.get("action_type")
                        del action["action_type"]
                    elif "action_type_str" not in action:
                        action["action_type_str"] = None

        next_task = final_state.get("next_task_details")
        if next_task:
            if ui_actions is None:
                ui_actions = []

            has_task_button = any(
                action.get("action_type_str") == "DISPLAY_NEXT_TASK_BUTTON"
                for action in ui_actions
            )

            if not has_task_button and next_task:
                mapped_action_type_str = "SHOW_ALERT"
                task_title = next_task.get("title", "Unknown Task")
                task_desc = next_task.get("description", "")
                mapped_parameters = {"message": f"Next Task: {task_title}\n{task_desc}"}

                ui_actions.append(
                    {
                        "action_type_str": mapped_action_type_str,
                        "parameters": mapped_parameters,
                    }
                )

        ui_actions_list: Optional[List[ReactUIAction]] = None
        if ui_actions:
            ui_actions_list = []
            for action_dict in ui_actions:
                action_type = action_dict.get("action_type_str", "")

                target_element_id = None
                if "target_element_id" in action_dict:
                    target_element_id = action_dict["target_element_id"]
                elif "targetElementId" in action_dict:
                    target_element_id = action_dict["targetElementId"]

                parameters = action_dict.get("parameters", {})

                ui_actions_list.append(
                    ReactUIAction(
                        action_type=action_type,
                        target_element_id=target_element_id,
                        parameters=parameters,
                    )
                )

        response = InteractionResponse(
            response=response_text, ui_actions=ui_actions_list
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Internal Server Error during AI processing"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    # You can add any connection logic here
    await sio.emit('welcome', {'data': 'Welcome to the Rox AI backend!'}, room=sid)

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@sio.event
async def message(sid, data):
    print(f"Message from {sid}: {data}")
    # Echo the message back to the client
    await sio.emit('response', {'response': f"Server received: {data}"}, room=sid)

# Helper function to convert text to audio base64
async def text_to_audio_base64(text: str) -> str:
    """Convert text to audio and return base64 encoded string"""
    try:
        # Create a bytes buffer to store the audio
        audio_buffer = io.BytesIO()
        
        # Generate audio from text using gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode audio to base64
        audio_data = audio_buffer.read()
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Return data URL format that can be directly used in an audio element
        return f"data:audio/mp3;base64,{base64_audio}"
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return ""

@sio.event
async def send_message(sid, data):
    print(f"Send message from {sid}: {data}")
    
    # Process the message using your existing AI logic if needed
    try:
        # Extract the message and check if audio is requested
        user_message = data.get('message', '')
        audio_requested = data.get('audio_response_requested', False)
        
        # Generate AI response (in a real app, this would call your AI model)
        ai_response = f"I understand your question about '{user_message}'. Here's what I can tell you: This is a simulated AI response that would normally contain relevant information about your query. I've analyzed your question and can provide detailed insights. For example, if you asked about course summary, I would give you statistics about your progress, areas of strength, and suggestions for improvement."
        
        # Send text response
        await sio.emit('ai_response', {
            'response': ai_response,
            'type': 'text'
        }, room=sid)
        
        # If audio is requested, generate and send audio response
        if audio_requested:
            # Split the response into smaller chunks to send separately
            # This simulates streaming audio in chunks
            sentences = ai_response.split('.')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    # Convert this sentence to audio
                    audio_data = await text_to_audio_base64(sentence + '.')
                    
                    # Send audio chunk to client
                    if audio_data:
                        await sio.emit('ai_audio', {
                            'audio': audio_data,
                            'chunk_index': i,
                            'total_chunks': len(sentences)
                        }, room=sid)
                        
                        # Small delay to simulate streaming
                        await asyncio.sleep(0.2)
    except Exception as e:
        print(f"Error processing message: {e}")
        await sio.emit('error', {'error': str(e)}, room=sid)

if __name__ == "__main__":
    import uvicorn
    import os

    # Use the Socket.IO app instead of just the FastAPI app
    uvicorn.run(socket_app, host="0.0.0.0", port=int(os.getenv("PORT", "5005")))
