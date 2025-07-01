import os
import threading
from typing import Optional, List, Dict, Any
from mem0 import Memory
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Singleton class for Mem0 client to avoid file lock issues on Windows
class Mem0Client:
    _instance: Optional['Mem0Client'] = None
    _lock = threading.Lock()
    is_initialized = False

    def __new__(cls, *args, **kwargs) -> 'Mem0Client':
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self.is_initialized:
            return
        with self._lock:
            if self.is_initialized:
                return
            
            logger.info("Initializing Mem0Client singleton...")
            self._initialize()
            self.is_initialized = True

    def _initialize(self):
        """Initializes the Mem0 instance using Google AI."""
        logger.info("--- [Mem0Client._initialize] START ---")
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY is required for Google AI provider.")

        config = {
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.0-flash",
                    "api_key": google_api_key,
                    "temperature": 0.7,
                }
            },
            "embedding": {
                "provider": "gemini",
                "config": {
                    "model": "text-embedding-004",
                    "api_key": google_api_key,
                }
            }
        }
        
        try:
            self.mem0_instance = Memory.from_config(config)
            logger.info("--- [Mem0Client._initialize] Initialized mem0_instance with Google AI config. ---")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 with Google AI config: {e}", exc_info=True)
            raise
        
        logger.info("--- [Mem0Client._initialize] END ---")

    def add(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug(f"Mem0Client: Calling add for user_id: {user_id} with messages: {messages} and metadata: {metadata}")
        try:
            # The `Memory` class expects the `data` argument.
            return self.mem0_instance.add(messages=messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Mem0Client: Error in add method: {e}", exc_info=True)
            raise

    def get_all(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get all memories for a user.
        The `Memory` class returns a list, but the rest of our app expects a dict.
        """
        logger.debug(f"Mem0Client: Calling get_all for user_id: {user_id}")
        try:
            raw_memories = self.mem0_instance.get_all(user_id=user_id)
            return {'results': raw_memories}
        except Exception as e:
            logger.error(f"Mem0Client: Error in get_all method: {e}", exc_info=True)
            raise

    def search(self, query: str, user_id: str, limit: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Search memories using semantic search.
        The `Memory` class returns a list, but the rest of our app expects a dict.
        """
        logger.debug(f"Mem0Client: Calling search with query: '{query}', user_id: '{user_id}'")
        try:
            results = self.mem0_instance.search(query=query, user_id=user_id, limit=limit)
            return {'results': results}
        except Exception as e:
            logger.error(f"Mem0Client: Error in search method: {e}", exc_info=True)
            raise

    def delete(self, memory_id: str) -> None:
        """Deletes a specific memory by its ID."""
        logger.debug(f"Mem0Client: Calling delete for memory_id: {memory_id}")
        try:
            self.mem0_instance.delete(memory_id=memory_id)
        except Exception as e:
            logger.error(f"Mem0Client: Error in delete method for memory_id {memory_id}: {e}", exc_info=True)
            raise

    def delete_all(self, user_id: str) -> None:
        """Deletes all memories for a specific user."""
        logger.info(f"Mem0Client: Deleting all memories for user_id: {user_id}")
        try:
            memories_response = self.get_all(user_id=user_id)
            memories_to_delete = memories_response.get('results', [])
            
            if not memories_to_delete:
                logger.info(f"Mem0Client: No memories found to delete for user_id: {user_id}")
                return

            for mem in memories_to_delete:
                self.delete(memory_id=mem.id)
            
            logger.info(f"Mem0Client: Successfully deleted all memories for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Mem0Client: Error in delete_all method for user_id {user_id}: {e}", exc_info=True)
            raise

# Create a single, shared instance of the client
logger.info("Creating shared_mem0_client instance.")
shared_mem0_client = Mem0Client()
logger.info("shared_mem0_client instance created.")
