import json
from state import AgentGraphState
import logging
import os
import google.generativeai as genai
from typing import Dict, AsyncIterator

logger = logging.getLogger(__name__)

async def conversation_handler_node(state: AgentGraphState) -> AsyncIterator[Dict]:
    """
    Handles the main conversational turn. It streams the response back to the client
    via 'streaming_text_chunk' and yields the final complete response to update the 
    graph state for subsequent nodes using 'conversational_tts' and 'output_content'.
    """
    logger.info(
        f"ConversationHandlerNode: Entry point activated for user {state.get('user_id', 'unknown_user')}"
    )
    api_key = os.getenv("GOOGLE_API_KEY")
    full_response_text = ""  # Accumulator for the full response

    try:
        logger.debug("Attempting genai.configure()...")
        genai.configure(api_key=api_key)
        logger.debug("genai.configure() successful.")

        logger.debug(
            f"Attempting to initialize GenerativeModel: gemini-1.5-flash-latest"
        )
        # Corrected model name and removed restrictive JSON mime type from original code
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        logger.debug("GenerativeModel initialized successfully.")

        llm_instruction = state.get("llm_instruction", "")
        user_data = state.get("user_data", {})
        transcript = state.get("transcript", "")

        logger.info(f"Retrieved llm_instruction from state: '{llm_instruction[:100]}...' if present else 'Not present'")
        logger.info(f"User data: {user_data}")
        logger.info(f"Transcript: '{transcript}'")

        # Get student memory context if available
        student_memory = state.get("student_memory_context")
        
        # Add null checking to prevent NoneType errors
        if student_memory is None:
            student_memory = {}
            logger.warning("Student memory context is None, using empty dictionary")
            
        profile = student_memory.get("profile", {})
        interaction_history = student_memory.get("interaction_history", [])
        
        # Use all available interaction history
        recent_interactions = []
        if interaction_history and isinstance(interaction_history, list):
            try:
                # Use all interactions from history
                recent_interactions = interaction_history
                logger.info(f"Retrieved {len(recent_interactions)} interactions from memory (using complete history)")
            except Exception as e:
                logger.warning(f"Error retrieving interaction history: {e}")
        
        if llm_instruction:
            # If explicit instruction is provided, use it directly
            full_prompt = llm_instruction
            logger.info(f"Using llm_instruction from state directly as full_prompt.")
        else:
            # Build a prompt that includes memory context
            system_prompt_text = """
            You are Rox, a friendly and encouraging AI guide for TOEFL preparation. Your primary goal is to create a highly personalized and adaptive learning experience by acting as a consistent tutor who remembers the student's entire journey.

            **Core Instructions for Personalization:**

            1.  **Remember Everything:** The provided memory contains the student's entire interaction history. Use it to recall past conversations, performance, preferences, and struggles.
            
            2.  **Identify and Address Patterns:** Pay close attention to recurring patterns. If you notice repeated mistakes (e.g., grammatical errors, spelling, specific concept misunderstandings), gently bring them up. Frame it constructively, like: "I've noticed we've worked on [topic] a few times. Let's try a different approach to really master it." or "That's a great point! It connects back to what we discussed about [previous topic]."

            3.  **Connect Past and Present:** Actively link current lessons to past ones. Make references like, "Remember when we talked about paragraph structure? You did really well with that. Let's apply the same thinking here." This shows you remember their progress and helps reinforce learning.

            4.  **Reference Student Traits:** Use the student's profile and conversation history to understand their personality and interests. If they mentioned liking a certain topic, try to connect it to the lesson. Acknowledge their feelings if they express frustration or excitement.

            5.  **Maintain a Consistent, Encouraging Tone:** Always be supportive. Your goal is to build the student's confidence. Start conversations with returning students with a warm "Welcome back!" to acknowledge your shared history.

            **Output Format:**
            - Respond directly with the text that should be spoken for text-to-speech.
            - Do NOT use JSON or any other special formatting in your final response.

            **Crucial Guideline on Accuracy:**
            - Ground all your observations and personalized references in the student's history provided below. 
            - While you should identify patterns and make connections, DO NOT invent specific events, facts, or conversations that are not present in the memory. If you don't have information about something, it's better to say you don't recall.

            The memory below contains ALL previous interactions. Use it to make your response as helpful and personal as possible.
            """
            
            # Add profile information if available
            profile_section = ""
            if profile:
                try:
                    profile_section = f"Student Profile: {json.dumps(profile, indent=2)}\n"
                    student_name = profile.get("name", "Student")
                    system_prompt_text += f"\nYou're speaking with {student_name}."
                except Exception as e:
                    logger.warning(f"Error formatting profile data: {e}")
            
            # Add recent interaction summary
            history_section = ""
            if recent_interactions:
                try:
                    # Format a comprehensive interaction history in a more readable format
                    history_section = "PREVIOUS CONVERSATIONS:\n\n"
                    conversation_count = 0
                    logger.info(f"Processing {len(recent_interactions)} interactions for memory context")
                    
                    for idx, interaction in enumerate(recent_interactions):
                        if isinstance(interaction, dict):
                            # First check for the structured memory format
                            student_msg = None
                            ai_response = None
                            task_details = None
                            
                            # Try multiple possible field names for the user message
                            if "transcript" in interaction:
                                student_msg = interaction.get("transcript")
                            elif "memory" in interaction:
                                student_msg = interaction.get("memory")
                            
                            # Try multiple possible field names for the AI response
                            if "assistant_response" in interaction:
                                ai_response = interaction.get("assistant_response")
                            elif "feedback" in interaction:
                                ai_response = interaction.get("feedback")
                            elif "content" in interaction and isinstance(interaction["content"], str):
                                # Some memories might store content directly
                                ai_response = interaction.get("content")
                            
                            # Check for task details
                            if "task_details" in interaction and interaction["task_details"]:
                                task_details = interaction["task_details"]
                                logger.info(f"Found task_details in memory: {task_details}")
                            
                            # If we found a user message and response, format it into the history
                            if student_msg:
                                conversation_count += 1
                                history_section += f"Conversation {conversation_count}:\n"
                                history_section += f"Student: {student_msg}\n"
                                
                                if ai_response:
                                    history_section += f"Your response: {ai_response}\n"
                                    
                                if task_details:
                                    history_section += f"Task details: {json.dumps(task_details)}\n"
                                    
                                history_section += "\n"
                                logger.debug(f"Added memory: {student_msg[:30]}...")
                    
                    logger.info(f"Built detailed conversation history with {conversation_count} formatted interactions from {len(recent_interactions)} total memories")
                except Exception as e:
                    logger.warning(f"Error formatting interaction history: {e}")
            
            # Combine all contexts with the current user query
            user_prompt_text = ""
            if profile_section:
                user_prompt_text += profile_section
            if history_section:
                user_prompt_text += history_section
            if user_data:
                user_prompt_text += f"Additional Context: {json.dumps(user_data)}\n"
            
            user_prompt_text += f"Current Query: {transcript}"
            full_prompt = f"{system_prompt_text}\n\n{user_prompt_text}"
            logger.info(f"Constructed enhanced prompt with memory context")
            logger.debug(f"Full prompt sections: profile={bool(profile_section)}, history={bool(history_section)}, user_data={bool(user_data)}")

        # Log a truncated version of the prompt for debugging
        truncated_prompt = full_prompt[:200] + "..." if len(full_prompt) > 200 else full_prompt
        logger.debug(f"Full prompt for LLM (truncated): {truncated_prompt}")
        logger.debug(f"Full prompt length: {len(full_prompt)} characters")

        try:
            logger.debug("Attempting model.generate_content_async(stream=True)...")
            async_stream = await model.generate_content_async(full_prompt, stream=True)
            logger.debug("model.generate_content_async(stream=True) call successful, iterating over stream...")
            
            streamed_something = False
            async for chunk in async_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    logger.debug(f"Streaming chunk text: {chunk.text}")
                    yield {"streaming_text_chunk": chunk.text}
                    full_response_text += chunk.text
                    streamed_something = True
                else:
                    logger.debug(f"Received a chunk without text or empty text: {chunk}")
            
            if not streamed_something:
                logger.warning("LLM stream completed without yielding any text chunks.")
                fallback_message = "I'm ready, but it seems I have no specific words for this moment!"
                yield {"streaming_text_chunk": fallback_message}
                full_response_text = fallback_message

        except Exception as gen_err:
            logger.error(
                f"Error during model.generate_content_async(stream=True): {gen_err}", exc_info=True
            )
            fallback_tts_chunk = "I'm having a little trouble thinking right now, but I'm here to help! (LLM Stream Error)"
            logger.info(
                f"ConversationHandlerNode: Yielding fallback TTS chunk due to stream generation error: {fallback_tts_chunk}"
            )
            yield {"streaming_text_chunk": fallback_tts_chunk}
            full_response_text = fallback_tts_chunk
        
        logger.info(f"ConversationHandlerNode: Finished streaming. Yielding final accumulated response to state.")
        yield {
            "conversational_tts": full_response_text,
            "output_content": full_response_text,
        }

    except Exception as e:
        logger.error(
            f"Outer error in ConversationHandlerNode (e.g., config, model init): {e}",
            exc_info=True,
        )
        fallback_tts_chunk = "Looks like I'm having a bit of trouble starting up. Let's try again in a moment. (LLM Setup Error)"
        logger.info(
            f"ConversationHandlerNode: Yielding fallback TTS due to setup error: {fallback_tts_chunk}"
        )
        yield {"streaming_text_chunk": fallback_tts_chunk}
        yield {
            "conversational_tts": fallback_tts_chunk,
            "output_content": fallback_tts_chunk,
        }
