import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Query, HTTPException

from .vector_store import VectorStore
from ..common.Database import Database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="",
    tags=["semantic_search"]
)

# Cache VectorStore instance
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get or create VectorStore instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

@router.get("/semantic_search")
async def global_semantic_search(
    query: str, 
    limit: int = Query(10, ge=1, le=100),
    fetch_full_messages: bool = Query(True),
    use_context: bool = Query(True)
) -> Dict[str, Any]:
    """
    Perform semantic search across all messages
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        fetch_full_messages: Whether to fetch full message objects from MongoDB
        use_context: Whether to use the context-aware search index
        
    Returns:
        Dictionary with search results
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Search for messages
        search_results = vector_store.search(
            query=query,
            limit=limit,
            use_context=use_context
        )
        
        # If no results, return empty list
        if not search_results:
            return {"results": [], "count": 0}
        
        # Fetch full messages if requested
        if fetch_full_messages:
            # Group message IDs by guild ID
            messages_by_guild = {}
            for result in search_results:
                guild_id = result.get("guild_id")
                message_id = result.get("message_id")
                if guild_id and message_id:
                    if guild_id not in messages_by_guild:
                        messages_by_guild[guild_id] = []
                    messages_by_guild[guild_id].append(message_id)
            
            # Fetch messages from each guild
            all_messages = {}
            for guild_id, message_ids in messages_by_guild.items():
                collection_messages = Database.get_guild_collection(guild_id, "messages")
                messages = list(collection_messages.find({"_id": {"$in": message_ids}}))
                for msg in messages:
                    all_messages[f"{guild_id}_{msg['_id']}"] = msg
            
            # Enrich search results with full message data
            for result in search_results:
                guild_id = result.get("guild_id")
                message_id = result.get("message_id")
                lookup_key = f"{guild_id}_{message_id}"
                if lookup_key in all_messages:
                    result["message"] = all_messages[lookup_key]
                
                # Include the full context text if we're using context-aware search
                if use_context and "content" in result:
                    # Extract full context from text field if it exists
                    result["full_context"] = result.get("content")
        
        return {
            "results": search_results,
            "count": len(search_results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")

@router.get("/guild/semantic_search")
async def semantic_search(
    guild_id: str, 
    query: str, 
    limit: int = Query(10, ge=1, le=100),
    fetch_full_messages: bool = Query(True),
    use_context: bool = Query(True)
) -> Dict[str, Any]:
    """
    Perform semantic search on guild messages
    
    Args:
        guild_id: Guild ID to search in
        query: Search query
        limit: Maximum number of results to return
        fetch_full_messages: Whether to fetch full message objects from MongoDB
        use_context: Whether to use the context-aware search index
        
    Returns:
        Dictionary with search results
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Search for messages (using global search and filtering results by guild_id)
        all_results = vector_store.search(
            query=query,
            limit=limit * 5,  # Request more results since we're filtering afterward
            use_context=use_context
        )
        
        # Filter results to only include the specified guild
        search_results = [result for result in all_results if result.get("guild_id") == guild_id]
        
        # Limit results after filtering
        search_results = search_results[:limit]
        
        # If no results, return empty list
        if not search_results:
            return {"results": [], "count": 0}
        
        # Fetch full messages if requested
        if fetch_full_messages:
            message_ids = [result["message_id"] for result in search_results]
            collection_messages = Database.get_guild_collection(guild_id, "messages")
            
            # Fetch messages
            messages = list(collection_messages.find({"_id": {"$in": message_ids}}))
            
            # Create a lookup by message_id
            message_lookup = {msg["_id"]: msg for msg in messages}
            
            # Enrich search results with full message data
            for result in search_results:
                message_id = result["message_id"]
                if message_id in message_lookup:
                    result["message"] = message_lookup[message_id]
                
                # Include the full context text if we're using context-aware search
                if use_context and "content" in result:
                    # Extract full context from text field if it exists
                    result["full_context"] = result.get("content")
        
        return {
            "results": search_results,
            "count": len(search_results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")

@router.post("/guild/build_index")
async def build_index(
    guild_id: str, 
    max_messages: int = Query(None),
    force_rebuild: bool = Query(False)
) -> Dict[str, Any]:
    """
    Build vector index for a guild
    
    Args:
        guild_id: Guild ID to index
        max_messages: Maximum number of messages to index (None = all)
        force_rebuild: Whether to force rebuild the index
        
    Returns:
        Dictionary with indexing results
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Build index
        result = vector_store.index_guild_messages(
            guild_id=guild_id,
            max_messages=max_messages
        )
        
        return {
            "status": "success",
            "indexed_messages": result["indexed_messages"],
            "processed_messages": result["processed_messages"],
            "elapsed_time": result["elapsed_time"]
        }
    
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise HTTPException(status_code=500, detail=f"Error building index: {str(e)}")

@router.get("/guild/semantic_search/count")
async def count_semantic_search(
    guild_id: str, 
    query: str,
    use_context: bool = Query(True)
) -> Dict[str, Any]:
    """
    Count semantic search results efficiently without retrieving messages
    
    Args:
        guild_id: Guild ID to search in
        query: Search query
        use_context: Whether to use the context-aware search index
        
    Returns:
        Dictionary with count of matching results
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Get count of matches
        count = vector_store.count_matches(
            query=query,
            guild_id=guild_id,
            use_context=use_context
        )
        
        return {
            "count": count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search count: {e}")
        raise HTTPException(status_code=500, detail=f"Error counting semantic search results: {str(e)}")
