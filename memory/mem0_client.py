import os
import threading
from typing import Optional, List, Dict, Any
from mem0 import MemoryClient
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
            
            self.api_key = os.getenv("MEM0_API_KEY")
            self.org_id = os.getenv("MEM0_ORG_ID")
            self.project_id = os.getenv("MEM0_PROJECT_ID")

            if not self.api_key:
                logger.error("MEM0_API_KEY not found in environment variables.")
                raise ValueError("MEM0_API_KEY not found in environment variables.")
            if not self.org_id:
                logger.error("MEM0_ORG_ID not found in environment variables.")
                raise ValueError("MEM0_ORG_ID not found in environment variables.")
            if not self.project_id:
                logger.error("MEM0_PROJECT_ID not found in environment variables.")
                raise ValueError("MEM0_PROJECT_ID not found in environment variables.")

            try:
                self.mem0_instance = MemoryClient(
                    api_key=self.api_key,
                    org_id=self.org_id,
                    project_id=self.project_id
                )
                logger.info("Mem0Client initialized successfully using MemoryClient with org_id and project_id.")
            except Exception as e:
                logger.error(f"Failed to initialize Mem0 client with MemoryClient: {e}", exc_info=True)
                raise
            
            self.is_initialized = True

    def add(self, messages: List[Dict[str, str]], user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug(f"Mem0Client: Calling add for user_id: {user_id} with messages: {messages} and metadata: {metadata}")
        try:
            return self.mem0_instance.add(messages=messages, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Mem0Client: Error in add method: {e}", exc_info=True)
            raise

    def get_all(self, user_id: str, page: int = 1, page_size: int = 100, output_format: str = 'v1.1') -> Dict[str, Any]:
        """Get all memories for a user with pagination support.
        
        Args:
            user_id (str): The user ID to get memories for
            page (int, optional): Page number for pagination. Defaults to 1.
            page_size (int, optional): Number of items per page. Defaults to 100.
            output_format (str, optional): Format version for the output. Defaults to 'v1.1'.
            
        Returns:
            Dict[str, Any]: Paginated memories or list of memories depending on the output format
        """
        logger.debug(f"Mem0Client: Calling get_all for user_id: {user_id}, page: {page}, page_size: {page_size}")
        try:
            # The latest Mem0 API supports pagination and output format
            raw_memories = self.mem0_instance.get_all(
                user_id=user_id, 
                page=page, 
                page_size=page_size, 
                output_format=output_format
            )
            
            # Handle different return formats for backward compatibility
            if isinstance(raw_memories, dict):
                if 'results' in raw_memories:
                    return raw_memories  # v1.1 format with pagination already included
                elif 'memories' in raw_memories:
                    # Newer format with different structure
                    return {'results': raw_memories['memories']}
            elif isinstance(raw_memories, list):
                # v1.0 format returns a list directly, wrap it
                return {'results': raw_memories}
                
            logger.warning(f"Mem0Client: get_all returned unexpected format: {type(raw_memories)}. Returning as is.")
            return raw_memories  # Return whatever we got

        except Exception as e:
            logger.error(f"Mem0Client: Error in get_all method: {e}", exc_info=True)
            raise

    def search(self, query: str, user_id: str = None, agent_id: str = None, limit: Optional[int] = None, 
              metadata: Optional[Dict[str, Any]] = None, categories: Optional[List[str]] = None,
              version: str = None, filters: Optional[Dict[str, Any]] = None,
              threshold: float = 0.1, output_format: str = 'v1.1') -> Dict[str, Any]:
        """Search memories using semantic search with enhanced filter options.
        
        Args:
            query (str): The search query
            user_id (str, optional): User ID to search memories for
            agent_id (str, optional): Agent ID to search memories for
            limit (int, optional): Max number of results to return
            metadata (Dict, optional): Metadata filter
            categories (List[str], optional): Categories filter
            version (str, optional): API version to use ('v1' or 'v2')
            filters (Dict, optional): Advanced filters for v2 search
            threshold (float, optional): Similarity threshold
            output_format (str, optional): Format version for output
            
        Returns:
            Dict[str, Any]: Search results
        """
        logger.debug(f"Mem0Client: Calling search with query: '{query}', user_id: '{user_id}', "
                   f"agent_id: '{agent_id}', filters: {filters}")
        try:
            # Build search parameters based on what's provided
            search_params = {
                'query': query,
                'threshold': threshold,
                'output_format': output_format
            }
            
            # Add optional parameters if provided
            if user_id:
                search_params['user_id'] = user_id
            if agent_id:
                search_params['agent_id'] = agent_id
            if limit:
                search_params['limit'] = limit
            if metadata:
                search_params['metadata'] = metadata
            if categories:
                search_params['categories'] = categories
            if version:
                search_params['version'] = version
            if filters:
                search_params['filters'] = filters
                
            return self.mem0_instance.search(**search_params)
        except Exception as e:
            logger.error(f"Mem0Client: Error in search method: {e}", exc_info=True)
            raise

# Create a single, shared instance of the client
logger.info("Creating shared_mem0_client instance.")
shared_mem0_client = Mem0Client()
logger.info("shared_mem0_client instance created.")
