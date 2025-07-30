import time
from typing import Optional
from uuid import uuid4

from camel.memories import ChatHistoryMemory
from camel.memories.context_creators.score_based import (
    ScoreBasedContextCreator,
)
from camel.memories.records import MemoryRecord
from camel.messages import BaseMessage
from camel.storages.key_value_storages.mem0_cloud import Mem0Storage
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.types import ModelType, OpenAIBackendRole, RoleType
from camel.utils import OpenAITokenCounter


class Mem0CloudToolkit(BaseToolkit):
    """A toolkit for interacting with Mem0 cloud memory storage.
    
    This toolkit provides methods for adding, retrieving, searching, and 
    deleting memories in Mem0's cloud storage system. Unlike the standard
    MemoryToolkit which works with local file-based memory, this toolkit
    provides cloud-based persistent memory with advanced search capabilities.
    
    Args:
        agent_id (str): The agent identifier for memory organization.
        user_id (str): The user identifier for memory organization.
        timeout (Optional[float], optional): Maximum execution time allowed for
            toolkit operations in seconds. If None, no timeout is applied.
            (default: :obj:`None`)
    """

    def __init__(
        self, 
        agent_id: str, 
        user_id: str, 
        timeout: Optional[float] = None
    ):
        super().__init__(timeout=timeout)
        self.agent_id = agent_id
        self.user_id = user_id
        self.memory = self._setup_memory()

    def _setup_memory(self) -> ChatHistoryMemory:
        """Sets up CAMEL's memory components with Mem0 as the backend."""
        storage = Mem0Storage(agent_id=self.agent_id, user_id=self.user_id)
        token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        context_creator = ScoreBasedContextCreator(
            token_counter=token_counter, token_limit=4096
        )
        return ChatHistoryMemory(
            context_creator=context_creator,
            storage=storage,
            agent_id=self.agent_id,
        )

    def add_memory(self, content: str, metadata: Dict = None) -> str:
        """Adds a new memory record to Mem0 cloud storage.

        Args:
            content (str): The information to remember.
            metadata (dict, optional): Additional data to store with the
                memory. Defaults to None.

        Returns:
            str: A confirmation message.
        """
        record = MemoryRecord(
            uuid=uuid4(),
            message=BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict=metadata or {},
                content=content,
            ),
            role_at_backend=OpenAIBackendRole.USER,
            timestamp=time.time(),
            agent_id=self.agent_id,
        )
        self.memory.write_records([record])
        return "Memory added successfully to Mem0 cloud."

    def retrieve_memories(self) -> str:
        """Retrieves all memories from Mem0 cloud storage.

        Returns:
            str: Raw API response as a string.
        """
        memories = self.memory._chat_history_block.storage.load()
        return f"Raw API Response:\n{str(memories)}"

    def search_memories(self, query: str) -> str:
        """Searches for memories in Mem0 cloud storage based on a query.

        This uses Mem0's semantic search with:
        - Vector-based semantic matching
        - Metadata filtering
        - Reranking for better results

        Args:
            query (str): The search term to find relevant memories.

        Returns:
            str: Raw API response as a string.
        """
        try:
            # Use OR filter for better matching - match if either user_id or agent_id matches
            filters = {
                "OR": [
                    {"user_id": self.user_id},
                    {"agent_id": self.agent_id}
                ]
            }
            
            results = self.memory._chat_history_block.storage.client.search(
                query,
                version="v2",
                filters=filters,
                output_format="v1.1",
                # Enable advanced features but with less restrictive settings
                rerank=True,           # Better result ordering
                threshold=0.01,        # Lower threshold for better recall
                top_k=10,             # Get more results
                keyword_search=True    # Enable keyword matching alongside semantic
            )
            return f"Raw API Response:\n{str(results)}"
        except Exception as e:
            return f"API Error: {str(e)}"

    def delete_memories(self) -> str:
        """Deletes all memories from Mem0 cloud storage for the current user and agent.

        Returns:
            str: A confirmation message.
        """
        self.memory.clear()
        return "All memories have been cleared from Mem0 cloud storage."

    def get_tools(self) -> list[FunctionTool]:
        """Returns a list of FunctionTool objects for the agent.
        
        Returns:
            list[FunctionTool]: List of memory management tools.
        """
        return [
            FunctionTool(self.add_memory),
            FunctionTool(self.retrieve_memories),
            FunctionTool(self.search_memories),
            FunctionTool(self.delete_memories),
        ]
