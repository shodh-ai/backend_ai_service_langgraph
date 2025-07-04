import logging
import json
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class KnowledgeClient:
    """A client to interact with the master curriculum map and knowledge base."""

    def __init__(self):
        """Initializes the client by loading the curriculum map from a JSON file."""
        logger.info("--- Initializing KnowledgeClient --- ")
        self._curriculum_map = self._load_curriculum_map()

    def _load_curriculum_map(self) -> Dict[str, Any]:
        """Loads the curriculum map from the specified JSON file."""
        try:
            # Construct the path relative to this file's location
            root_dir = Path(__file__).resolve().parent.parent
            map_path = root_dir / 'knowledge_base_content' / 'curriculum_map.json'
            logger.info(f"Loading curriculum map from: {map_path}")
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} learning objectives from curriculum map.")
            return data
        except FileNotFoundError:
            logger.error(f"CRITICAL: Curriculum map file not found at {map_path}. The system will not have planning capabilities.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"CRITICAL: Failed to parse JSON from curriculum map file at {map_path}.")
            return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the curriculum map: {e}", exc_info=True)
            return {}

    async def get_full_curriculum_map(self) -> Dict[str, Any]:
        """Fetches the entire curriculum map of Learning Objectives."""
        logger.debug("KnowledgeClient: Fetching full curriculum map.")
        return self._curriculum_map

# Create a single, shared instance of the client for the application to use.
knowledge_client = KnowledgeClient()
