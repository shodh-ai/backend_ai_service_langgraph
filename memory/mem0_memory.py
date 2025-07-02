import logging
import pickle
import base64
import threading
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from .mem0_client import shared_mem0_client

logger = logging.getLogger(__name__)

load_dotenv()

class StudentProfileMemory:
    """
    Handles application-specific memory operations like student profiles and interactions.
    """
    def __init__(self):
        logger.info("--- [StudentProfileMemory.__init__] START ---")
        self.mem0_client = shared_mem0_client
        self.user_id_field = 'user_id'
        logger.info("--- [StudentProfileMemory.__init__] Using shared Mem0 client. ---")
        logger.info("--- [StudentProfileMemory.__init__] END ---")

    def add_interaction(self, user_id: str, interaction_summary: Dict[str, Any]) -> None:
        """Add an interaction summary to memory."""
        logger.info(f"StudentProfileMemory: Attempting to add interaction (summary): {interaction_summary} for user_id: {user_id}")
        try:
            # Format the interaction content
            interaction_content_str = json.dumps(interaction_summary)
            formatted_message = [{"role": "user", "content": interaction_content_str}]
            
            self.mem0_client.add(
                messages=formatted_message, 
                user_id=user_id, 
                metadata={'type': 'interaction', self.user_id_field: user_id}
            )
            logger.info(f"StudentProfileMemory: Successfully added interaction for user_id: {user_id}")
        except Exception as e:
            logger.error(f"StudentProfileMemory: ERROR during self.mem0_instance.add() for user_id: {user_id}. Exception: {e}", exc_info=True)
            raise

    def update_student_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Updates or sets profile data for a student using mem0."""
        logger.info(f"StudentProfileMemory: Attempting to update profile for user_id: {user_id} with data: {profile_data}")
        try:
            profile_data_str = json.dumps(profile_data)
            # Use 'system' role to distinguish profile data from user interactions
            formatted_message = [{"role": "system", "content": profile_data_str}]
            
            self.mem0_client.add(
                messages=formatted_message,
                user_id=user_id, 
                metadata={'type': 'profile', self.user_id_field: user_id}
            )
            logger.info(f"StudentProfileMemory: Updated/added profile for {user_id}.")
        except Exception as e:
            logger.error(f"StudentProfileMemory: Error updating profile for {user_id}: {e}", exc_info=True)
            raise

    def get_student_data(self, user_id: str) -> Dict[str, Any]:
        """Get all student data including profile and interactions."""
        logger.info(f"StudentProfileMemory: Getting student data for user_id: {user_id}")
        try:
            response = self.mem0_client.get_all(user_id=user_id)
            all_memories = response.get('results', [])
            
            profile_data = {}
            interactions = []
            
            for mem in all_memories:
                content_str = mem.get('text')
                if not content_str:
                    continue

                try:
                    content_dict = json.loads(content_str)
                    if mem.get('metadata', {}).get('type') == 'profile':
                        if isinstance(content_dict, dict):
                            profile_data.update(content_dict)
                    elif mem.get('metadata', {}).get('type') == 'interaction':
                        interactions.append(content_dict)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse content for memory {mem.get('id')} for user {user_id}")
            
            return {
                'profile': profile_data,
                'interactions': interactions,
                'total_memories': len(all_memories)
            }
        except Exception as e:
            logger.error(f"StudentProfileMemory: Error getting student data for {user_id}: {e}", exc_info=True)
            return {'profile': {}, 'interactions': [], 'total_memories': 0}

    def clear_user_memory(self, user_id: str) -> None:
        """Clears all memory for a specific user in mem0."""
        logger.info(f"StudentProfileMemory: Attempting to clear all data for user_id: {user_id}")
        try:
            self.mem0_client.delete_all(user_id=user_id)
            logger.info(f"StudentProfileMemory: Cleared all data for user_id: {user_id}")
        except Exception as e:
            logger.error(f"StudentProfileMemory: Error clearing memory for {user_id}: {e}", exc_info=True)
            raise

class Mem0Checkpointer(BaseCheckpointSaver):
    """A LangGraph checkpointer that stores the entire graph state in Mem0."""

    def __init__(self, **kwargs):
        mem0_client = kwargs.pop("mem0_client", shared_mem0_client)
        super().__init__(serde=pickle, **kwargs)
        self.mem0_client = mem0_client
        self.user_id_field = 'user_id'

    def get(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        """Get checkpoint by config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        response = self.mem0_client.get_all(user_id=thread_id)
        memories = response.get('results', [])
        checkpoints = [m for m in memories if m.get('metadata', {}).get("type") == "langgraph_checkpoint"]

        if not checkpoints:
            return None

        if checkpoint_id:
            target_memory = next((m for m in checkpoints if m.get('id') == checkpoint_id), None)
        else:
            # Return the latest checkpoint if no ID is specified
            target_memory = max(checkpoints, key=lambda x: x.get('metadata', {}).get("version_ts", "0"))

        if not target_memory:
            return None

        try:
            # Decode from base64 and then unpickle
            decoded_bytes = base64.b64decode(target_memory.get('text').encode('utf-8'))
            return self.serde.loads(decoded_bytes)
        except Exception as e:
            logger.error(f"Error deserializing checkpoint: {e}")
            return None

    def put(self, config: RunnableConfig, checkpoint: Dict[str, Any]) -> RunnableConfig:
        """Store checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        
        # The 'version' is the checkpoint_id we will return
        version = self.get_next_version(None)
        
        try:
            # Pickle the checkpoint and then encode it in base64
            pickled_checkpoint = self.serde.dumps(checkpoint)
            base64_encoded_checkpoint = base64.b64encode(pickled_checkpoint).decode('utf-8')

            # Save to mem0
            created_memory = self.mem0_client.add(
                messages=[{"role": "system", "content": base64_encoded_checkpoint}],
                user_id=thread_id,
                metadata={
                    "type": "langgraph_checkpoint",
                    "version_ts": version
                }
            )
            
            # The client's add method returns a dict, we need the ID from it.
            memory_id = created_memory.get('id')

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": memory_id,
                }
            }
        except Exception as e:
            logger.error(f"Error storing checkpoint: {e}")
            raise

    def list(self, config: RunnableConfig) -> List[RunnableConfig]:
        """List all checkpoints for a thread."""
        thread_id = config["configurable"]["thread_id"]
        response = self.mem0_client.get_all(user_id=thread_id)
        memories = response.get('results', [])
        checkpoints = [m for m in memories if m.get('metadata', {}).get("type") == "langgraph_checkpoint"]

        return [
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": mem.get('id'),
                },
                "metadata": {
                    "source": "mem0",
                    "timestamp": mem.get('metadata', {}).get("version_ts"),
                },
            }
            for mem in checkpoints
        ]

    def get_next_version(self, current_version: Optional[Union[int, str]]) -> Union[int, str]:
        """Generate next version timestamp."""
        return str(datetime.now(timezone.utc).timestamp())

    async def aget(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        """Async version of get."""
        return self.get(config)

    async def aput(self, config: RunnableConfig, checkpoint: Dict[str, Any]) -> RunnableConfig:
        """Async version of put."""
        return self.put(config, checkpoint)

    async def alist(self, config: RunnableConfig) -> List[RunnableConfig]:
        """Async version of list."""
        return self.list(config)
