import logging
import json
from datetime import datetime
from typing import Any, Dict

from state import AgentGraphState
from memory.mem0_client import shared_mem0_client

logger = logging.getLogger(__name__)

async def load_student_data_node(state: AgentGraphState) -> dict:
    """
    Loads student data from Mem0, extracts 'next_task_details' from the most recent
    interaction, and updates the state.
    """
    user_id = state["user_id"]
    logger.info(f"StudentModelNode: Loading student data for user_id: '{user_id}' from Mem0")

    # Get all memories from Mem0 for the user
    try:
        all_memories = shared_mem0_client.get_all(user_id=user_id)
        student_data: Dict[str, Any] = {"profile": {}, "interaction_history": []}
        
        # Handle different possible formats returned by Mem0
        memories_list = []
        try:
            if isinstance(all_memories, dict):
                if 'results' in all_memories:
                    memories_list = all_memories['results']
                    
                    # Handle cases where 'results' might be a dict containing another 'results' list
                    if isinstance(memories_list, dict) and 'results' in memories_list:
                        logger.warning("Nested 'results' key found in Mem0 response. Extracting inner list.")
                        memories_list = memories_list['results']

                    if not isinstance(memories_list, list):
                        logger.warning(f"'results' is not a list but {type(memories_list).__name__}. Converting to list.")
                        memories_list = [memories_list] if memories_list is not None else []
                    logger.info(f"Memory format: standard v1.1 results format with {len(memories_list)} memories")
                elif 'memories' in all_memories:
                    memories_list = all_memories['memories']
                    if not isinstance(memories_list, list):
                        logger.warning(f"'memories' is not a list but {type(memories_list).__name__}. Converting to list.")
                        memories_list = [memories_list] if memories_list is not None else []
                    logger.info(f"Memory format: newer API format with {len(memories_list)} memories")
                else:
                    logger.warning(f"Unknown dict format from Mem0 client: {list(all_memories.keys())[:5] if all_memories else 'empty dict'}")
                    # Try to use the dictionary itself as a single memory if it's not empty
                    if all_memories: 
                        memories_list = [all_memories]
            elif isinstance(all_memories, list):
                memories_list = all_memories
                logger.info(f"Memory format: plain list format with {len(memories_list)} memories")
            elif all_memories is not None:
                logger.warning(f"Unexpected format from Mem0 client: {type(all_memories)}")
                # Try to use the object itself as a single memory
                memories_list = [all_memories]
            else:
                logger.warning("Memory result is None from Mem0 client")
                memories_list = [] # Explicitly set to empty list
        except Exception as e:
            logger.error(f"Error processing Mem0 response format: {str(e)}")
            memories_list = [] # Ensure memories_list is an empty list on error
            
        # Log memory breakdown for debugging with detailed structure analysis
        memory_types = {}
        
        # Detailed memory inspection
        logger.info(f"DETAILED MEMORY INSPECTION FOR USER {user_id}:")
        for i, mem in enumerate(memories_list[:5]):  # Limit to first 5 to avoid excessive logs
            try:
                logger.info(f"Memory #{i+1} Type: {type(mem).__name__}")
                
                # Examine top-level structure
                if isinstance(mem, dict):
                    logger.info(f"Memory #{i+1} Keys: {list(mem.keys())}")
                    
                    # Look for content in different possible locations
                    if 'data' in mem:
                        logger.info(f"Memory #{i+1} has 'data' field of type: {type(mem['data']).__name__}")
                        if isinstance(mem['data'], dict):
                            logger.info(f"Memory #{i+1} data keys: {list(mem['data'].keys())[:5]}")
                    
                    if 'metadata' in mem:
                        logger.info(f"Memory #{i+1} has 'metadata' field of type: {type(mem['metadata']).__name__}")
                        if isinstance(mem['metadata'], dict):
                            logger.info(f"Memory #{i+1} metadata: {mem['metadata']}")
                            
                    if 'messages' in mem:
                        logger.info(f"Memory #{i+1} has 'messages' field with {len(mem['messages'])} messages")
                        # Check first message if available
                        if mem['messages'] and isinstance(mem['messages'], list) and len(mem['messages']) > 0:
                            logger.info(f"Memory #{i+1} first message: {mem['messages'][0]}")
                    
                    # Special checks for content fields
                    for field in ['content', 'transcript', 'text', 'user_input', 'memory']:
                        if field in mem and mem[field]:
                            logger.info(f"Memory #{i+1} has direct '{field}' field with content: {str(mem[field])[:50]}...")
                            
                    # Check for potential Mem0 v1.1 structure
                    if 'id' in mem and 'user_id' in mem:
                        logger.info(f"Memory #{i+1} appears to be a Mem0 v1.1 memory with ID: {mem['id']}")
                
                # Try to get memory type for statistics
                mem_type = 'unknown'
                if hasattr(mem, 'metadata') and hasattr(mem.metadata, 'get'):
                    mem_type = mem.metadata.get('type', 'unknown')
                elif isinstance(mem, dict) and 'metadata' in mem and isinstance(mem['metadata'], dict):
                    mem_type = mem['metadata'].get('type', 'unknown')
                    
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                
            except Exception as e:
                logger.error(f"Error inspecting memory #{i+1}: {e}")
                memory_types['error'] = memory_types.get('error', 0) + 1
                
        logger.info(f"Memory type breakdown: {memory_types}")
            
        # Process all memories regardless of metadata type
        for mem in memories_list:
            # Handle different memory formats
            try:
                # Extract metadata - could be an attribute or a dictionary key
                meta = {}
                if hasattr(mem, 'metadata'):
                    meta = mem.metadata or {}
                elif isinstance(mem, dict) and 'metadata' in mem:
                    meta = mem['metadata'] or {}
                    
                # Extract data from multiple possible locations
                mem_data = None
                content = None
                transcript = None
                assistant_response = None
                memory_type = meta.get('type', 'unknown')
                
                # 1. Try to get data from the data field
                if hasattr(mem, 'data'):
                    mem_data = mem.data
                elif isinstance(mem, dict) and 'data' in mem:
                    mem_data = mem['data']
                else:
                    # If we can't find a data field, use the memory itself as data
                    mem_data = mem
                    
                # 2. Try to extract content from various possible fields
                if isinstance(mem_data, dict):
                    # For dictionaries, look for content fields
                    if 'transcript' in mem_data:
                        transcript = mem_data.get('transcript')
                    elif 'content' in mem_data:
                        content = mem_data.get('content')
                    elif 'text' in mem_data:
                        content = mem_data.get('text')
                    
                    # Look for assistant response fields
                    if 'assistant_response' in mem_data:
                        assistant_response = mem_data.get('assistant_response')
                    elif 'feedback' in mem_data:
                        assistant_response = mem_data.get('feedback')
                    elif 'response' in mem_data:
                        assistant_response = mem_data.get('response')
                    
                # For direct content access in the memory object
                if isinstance(mem, dict):
                    if not transcript and 'transcript' in mem:
                        transcript = mem.get('transcript')
                    if not content and 'content' in mem:
                        content = mem.get('content')
                    if not assistant_response and 'assistant_response' in mem:
                        assistant_response = mem.get('assistant_response')
                
                # Determine what to do with the extracted data based on memory type
                if memory_type == 'profile':
                    # Handle profile data
                    if isinstance(mem_data, dict):
                        student_data["profile"].update(mem_data)
                    elif isinstance(mem_data, str):
                        try:
                            profile_dict = json.loads(mem_data)
                            if isinstance(profile_dict, dict):
                                student_data["profile"].update(profile_dict)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Could not parse profile memory for user {user_id}: {mem_data}")
                
                elif memory_type == 'structured_interaction':
                    # Special handling for structured_interaction memories
                    try:
                        logger.info(f"Processing structured_interaction memory of type: {mem_data.__class__.__name__}")
                        
                        extracted_structured_data = None
                        messages = []
                        
                        # Extract messages from various formats
                        if isinstance(mem_data, dict) and 'messages' in mem_data:
                            messages = mem_data['messages']
                            logger.info(f"Found {len(messages)} messages in 'messages' dictionary field")
                        elif isinstance(mem_data, list):
                            messages = mem_data
                            logger.info(f"Found {len(messages)} messages in direct list")
                        elif isinstance(mem, dict) and 'messages' in mem:
                            # Sometimes the messages are directly in the memory object
                            messages = mem['messages']
                            logger.info(f"Found {len(messages)} messages directly in memory object")
                        
                        # Process messages if we found any
                        if messages:
                            for msg in messages:
                                if isinstance(msg, dict) and msg.get('role') == 'system' and 'content' in msg:
                                    try:
                                        content_str = msg['content']
                                        logger.info(f"Attempting to parse system message content: {content_str[:50]}...")
                                        structured_data = json.loads(content_str)
                                        
                                        if isinstance(structured_data, dict):
                                            logger.info("Successfully extracted structured data from message content")
                                            extracted_structured_data = structured_data
                                            break
                                    except (json.JSONDecodeError, TypeError) as e:
                                        logger.warning(f"Failed to parse message content: {e}")
                        
                        # Handle direct dictionary data if no messages processed
                        if not extracted_structured_data and isinstance(mem_data, dict):
                            logger.info("No structured data from messages, using memory data directly")
                            extracted_structured_data = mem_data
                            
                        # Handle data extraction from Mem0 v1.1 'content' field if present
                        if not extracted_structured_data and isinstance(mem, dict) and 'content' in mem:
                            try:
                                if isinstance(mem['content'], str):
                                    content_data = json.loads(mem['content'])
                                    if isinstance(content_data, dict):
                                        extracted_structured_data = content_data
                                        logger.info("Extracted structured data from direct 'content' field")
                                elif isinstance(mem['content'], dict):
                                    extracted_structured_data = mem['content']
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.warning(f"Failed to parse direct content field: {e}")
                                                
                        # Process extracted data
                        if extracted_structured_data:
                            # Extract task details if present
                            if 'task_details' in extracted_structured_data and extracted_structured_data['task_details']:
                                task_details = extracted_structured_data['task_details']
                                logger.info(f"Found task_details in structured data: {task_details}")
                                next_task_details = task_details
                            
                            # Add to interaction history for conversation recall
                            student_data["interaction_history"].append(extracted_structured_data)
                            logger.info(f"Added structured interaction to history with keys: {list(extracted_structured_data.keys())}")
                        else:
                            logger.warning("Could not extract any structured data")
                    
                    except Exception as e:
                        logger.error(f"Error processing structured interaction: {str(e)}", exc_info=True)
                
                # General case: Handle any other type of interaction memory
                else:
                    # Create interaction entry
                    interaction = {}
                    
                    # Add transcript/content if available
                    if transcript:
                        interaction['transcript'] = transcript
                    elif content:
                        interaction['content'] = content
                        
                    # Add assistant response if available
                    if assistant_response:
                        interaction['assistant_response'] = assistant_response
                        
                    # Use full memory data if nothing specific was found
                    if not interaction and isinstance(mem_data, dict):
                        interaction = mem_data
                    elif not interaction:
                        # Last resort: try to parse as JSON if it's a string
                        if isinstance(mem_data, str):
                            try:
                                parsed_data = json.loads(mem_data)
                                if isinstance(parsed_data, dict):
                                    interaction = parsed_data
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(f"Could not parse memory as JSON: {mem_data[:50]}...")
                    
                    # Only add if we found something useful
                    if interaction:
                        student_data["interaction_history"].append(interaction)
                        logger.debug(f"Added memory to interaction history: {str(interaction)[:50]}...")
                    else:
                        logger.warning(f"Could not extract useful interaction data from memory: {type(mem_data)}")
                
            except Exception as inner_e:
                logger.warning(f"Error processing individual memory: {inner_e}. Skipping this memory.")
                continue

        # Sort history just in case, though mem0 usually returns latest first
        student_data["interaction_history"].sort(key=lambda x: x.get('timestamp', 0), reverse=True)

    except Exception as e:
        logger.error(f"Failed to retrieve or process data from Mem0 for user {user_id}: {e}", exc_info=True)
        student_data = {"profile": {}, "interaction_history": []}
    logger.info(f"StudentModelNode: Retrieved student data from Mem0: {student_data}")

    # Initialize updates with the full student memory context
    updates = {"student_memory_context": student_data}

    # Extract 'next_task_details' from the last interaction, if available
    interaction_history = (student_data or {}).get("interaction_history", [])
    if interaction_history:
        last_interaction = interaction_history[-1]
        next_task_details = last_interaction.get("task_details")
        if next_task_details:
            updates["next_task_details"] = next_task_details
            logger.info(f"StudentModelNode: Loaded 'next_task_details' from last interaction: {next_task_details}")
        else:
            logger.info("StudentModelNode: Last interaction found, but it has no 'task_details'.")
    else:
        logger.info("StudentModelNode: No interaction history found for user, so no 'next_task_details' to load.")

    return updates

async def save_interaction_node(state: AgentGraphState) -> dict:
    """Saves the current interaction to Mem0."""
    user_id = state["user_id"]
    
    # Get key data from state
    transcript = state.get("transcript", "")
    output_content = state.get("output_content") or state.get("feedback_content", "")

    # Prepare interaction summary for logging
    interaction_data = {
        "transcript": transcript,
        "full_submitted_transcript": state.get("full_submitted_transcript"),
        "diagnosis": state.get("diagnosis_result"),
        "feedback": output_content.get("text_for_tts") if isinstance(output_content, dict) else output_content,
        "task_details": state.get("next_task_details")
    }
    
    logger.info(f"StudentModelNode: Saving interaction for user_id: '{user_id}' to Mem0")
    logger.info(f"StudentModelNode: All keys in state: {list(state.keys())}")
    raw_initial_report_from_state = state.get("initial_report_content") # Changed to use initial_report_content
    initial_report_content_from_state = state.get("initial_report_content") # For comparison
    logger.info(f"StudentModelNode: Value of 'initial_report_content' (as raw_initial_report_from_state) from state: {str(raw_initial_report_from_state)[:500]} (Type: {type(raw_initial_report_from_state)})")
    logger.info(f"StudentModelNode: Value of 'initial_report_content' from state: {str(initial_report_content_from_state)[:500]} (Type: {type(initial_report_content_from_state)})")
    logger.info("---------------- CHECKPOINT: EXECUTING MODIFIED student_model_node.py ----------------") # New prominent log
    logger.debug(f"StudentModelNode: Interaction data: {interaction_data}")
    
    # Format data for Mem0, ensuring content is a string
    # Consolidate assistant's message for Mem0, including pedagogical reasoning if available
    raw_report = state.get("initial_report_content") # Changed to use initial_report_content
    if raw_report and isinstance(raw_report, dict):
        assistant_content = json.dumps(raw_report, indent=2)
        logger.info("StudentModelNode: Using raw_initial_report_output for Mem0 assistant_content.")
    else:
        assistant_content = state.get("final_text_for_tts")
        logger.info(f"StudentModelNode: 'initial_report_content' (as raw_report) not found or not a dict. Falling back to final_text_for_tts for Mem0. Type: {type(raw_report)}")
    raw_pedagogy_output = state.get("raw_pedagogy_output")
    logger.info(f"StudentModelNode: Retrieved raw_pedagogy_output from state: {raw_pedagogy_output}")
    logger.info(f"StudentModelNode: Type of raw_pedagogy_output: {type(raw_pedagogy_output)}")

    # Ensure final_text_for_tts is a string
    assistant_tts_str = str(assistant_content) if assistant_content is not None else ""

    pedagogy_reasoning_str = ""
    if isinstance(raw_pedagogy_output, dict):
        pedagogy_reasoning = raw_pedagogy_output.get("reasoning")
        if pedagogy_reasoning is not None:
            pedagogy_reasoning_str = str(pedagogy_reasoning)
        logger.info(f"StudentModelNode: Extracted pedagogy_reasoning: {pedagogy_reasoning}")
    logger.info(f"StudentModelNode: pedagogy_reasoning_str after extraction: '{pedagogy_reasoning_str}'")

    if pedagogy_reasoning_str.strip():
        # Combine TTS and reasoning
        if assistant_tts_str.strip():
            assistant_content = f"{assistant_tts_str}\n\nPedagogical Reasoning: {pedagogy_reasoning_str}"
        else:
            # Only reasoning is present
            assistant_content = f"Pedagogical Reasoning: {pedagogy_reasoning_str}"
    elif assistant_tts_str.strip():
        # Only TTS is present
        assistant_content = assistant_tts_str
    else:
        # Neither is present or both are empty/whitespace
        assistant_content = "Assistant did not provide textual output for this interaction."
    logger.info(f"StudentModelNode: Final assistant_content for Mem0: '{assistant_content}'")
    messages_to_save = []
    # Only add user message if transcript is not empty
    if transcript and transcript.strip():
        messages_to_save.append({"role": "user", "content": transcript})

    # Only add assistant message if content is not empty
    if assistant_content and assistant_content.strip():
        messages_to_save.append({"role": "assistant", "content": assistant_content})

    # Extract next_task_details for saving to memory
    next_task_details = state.get("next_task_details")
    logger.info(f"StudentModelNode: Saving next_task_details to Mem0: {next_task_details}")
    
    # Prepare enhanced metadata with task_details if available
    memory_metadata = {'type': 'interaction'}
    if next_task_details:
        memory_metadata['task_details'] = next_task_details
    
    # Also create a structured memory that includes task_details for better retrieval
    structured_memory_data = None
    if transcript and transcript.strip():
        structured_memory_data = {
            "transcript": transcript,
            "assistant_response": assistant_content,
            "task_details": next_task_details,
            "timestamp": datetime.now().isoformat()
        }
    
    # Save to Mem0 only if there are messages to save
    if messages_to_save:
        try:
            # First save the conversation messages
            shared_mem0_client.add(
                messages=messages_to_save,
                user_id=user_id,
                metadata=memory_metadata
            )
            
            # Then save structured memory separately if available
            if structured_memory_data:
                try:
                    # According to Mem0 docs, we need to structure as a message with role and content
                    structured_messages = [
                        {
                            "role": "system", 
                            "content": json.dumps(structured_memory_data)  # Convert to JSON string as content
                        }
                    ]
                    # Add with proper metadata to distinguish this type of memory
                    shared_mem0_client.add(
                        messages=structured_messages,
                        user_id=user_id,
                        metadata={'type': 'structured_interaction', 'contains_task_details': True}
                    )
                    logger.info("Saved structured interaction memory with task_details to Mem0")
                except Exception as structured_err:
                    logger.warning(f"Failed to save structured memory: {structured_err}")
            
            logger.info(f"Successfully saved interaction for user_id: '{user_id}' to Mem0.")
        except Exception as e:
            logger.error(f"Failed to save interaction to Mem0 for user_id: '{user_id}': {e}", exc_info=True)
    else:
        logger.warning(f"No content to save to Mem0 for user_id: '{user_id}'. Skipping save.")
    
    return {} # No direct state update needed
