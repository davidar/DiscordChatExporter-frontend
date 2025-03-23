from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging
import uuid

from ..vector_search.vector_store import VectorStore, COLLECTION_NAME
from ..vector_search.topic_clustering import TopicClusterer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from ..messages.get_messages import get_messages_cursor_pagination

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="",
    tags=["guild"]
)

@router.get("/guild/semantic_distances")
async def get_semantic_distances(
    guild_id: str,
    channel_id: str,
    prev_page_cursor: Optional[str] = None,
    next_page_cursor: Optional[str] = None,
    around_page_cursor: Optional[str] = None,
    limit: int = 50,
):
    """
    Get semantic distances between adjacent messages in a channel
    
    This calculates cosine distances between message embeddings to identify
    potential topic boundaries in the conversation
    """
    try:
        # Initialize vector store to access embeddings
        vector_store = VectorStore()
        
        # Use the same pagination logic as messages endpoint
        message_data = await get_messages_cursor_pagination(
            guild_id=guild_id,
            channel_id=channel_id,
            prev_page_cursor=prev_page_cursor,
            next_page_cursor=next_page_cursor,
            around_page_cursor=around_page_cursor,
            limit=limit
        )
        
        # Extract messages
        messages = message_data.get("messages", [])
        
        if len(messages) < 2:
            # Not enough messages to calculate distances
            return {
                "prev_page_cursor": message_data.get("prev_page_cursor"),
                "messageDistances": [],
                "next_page_cursor": message_data.get("next_page_cursor")
            }
        
        # Need to fetch message vectors from the vector store
        message_ids = [msg["_id"] for msg in messages]
        
        # Debug logging
        logger.info(f"Fetching vectors for message_ids: {message_ids[:5]}...")
        
        # Convert message IDs to UUIDs in exactly the same way as vector_store.py
        uuid_ids = []
        id_mapping = {}  # Maps UUID back to original message ID
        
        for msg_id in message_ids:
            # Create deterministic UUID from message ID - EXACT SAME FORMAT as in vector_store.py
            combined_id = f"{guild_id}_{msg_id}"
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined_id))
            uuid_ids.append(doc_id)
            id_mapping[doc_id] = msg_id
        
        logger.info(f"First few UUID conversions: {list(zip(message_ids[:3], uuid_ids[:3]))}")
        
        # Fetch vectors for these messages
        try:
            # Use the Qdrant client to get vectors for all messages at once
            query_results = vector_store.qdrant_client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=uuid_ids,
                with_vectors=True
            )
            
            logger.info(f"Retrieved {len(query_results)} vectors from Qdrant")
        except Exception as qdrant_error:
            logger.error(f"Error retrieving vectors from Qdrant: {qdrant_error}")
            raise HTTPException(status_code=500, detail=f"Vector retrieval error: {str(qdrant_error)}")
        
        # Create lookup table for message ID -> vector
        vector_lookup = {}
        for point in query_results:
            if point.vector is not None:
                # Map back to original message ID
                original_id = id_mapping.get(str(point.id))
                if original_id:
                    vector_lookup[original_id] = np.array(point.vector)
        
        logger.info(f"Built vector lookup with {len(vector_lookup)} entries")
        
        # Calculate distances between adjacent messages
        message_distances = []
        
        for i in range(len(messages) - 1):
            current_msg_id = messages[i]["_id"]
            next_msg_id = messages[i + 1]["_id"]
            
            # Check if both messages have vectors
            if current_msg_id in vector_lookup and next_msg_id in vector_lookup:
                # Get vectors
                current_vector = vector_lookup[current_msg_id]
                next_vector = vector_lookup[next_msg_id]
                
                # Normalize vectors
                current_vector_norm = normalize(current_vector.reshape(1, -1))[0]
                next_vector_norm = normalize(next_vector.reshape(1, -1))[0]
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    current_vector_norm.reshape(1, -1),
                    next_vector_norm.reshape(1, -1)
                )[0][0]
                
                # Convert to distance (1 - similarity)
                distance = 1.0 - similarity
                
                message_distances.append({
                    "messageId": current_msg_id,
                    "nextMessageId": next_msg_id,
                    "distance": float(distance)
                })
        
        logger.info(f"Calculated {len(message_distances)} message distances")
        
        return {
            "prev_page_cursor": message_data.get("prev_page_cursor"),
            "messageDistances": message_distances,
            "next_page_cursor": message_data.get("next_page_cursor")
        }
        
    except Exception as e:
        logger.error(f"Error getting semantic distances: {e}")
        raise HTTPException(status_code=500, detail=str(e))
