import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
import tqdm
import pathlib
import hashlib
import json

# HuggingFace for embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder

# Qdrant for vector storage
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# MongoDB access
from ..common.Database import Database

# Constants
COLLECTION_NAME = "discord_messages"
CONTEXT_COLLECTION_NAME = "discord_messages_with_context"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Optimized for reranking
VECTOR_DIMENSION = 384  # BGE-small-en-v1.5 dimensions
DEFAULT_BATCH_SIZE = 1024
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_PRECEDING_MESSAGES = 5  # Maximum number of preceding messages to include
MAX_REPLY_CHAIN_LENGTH = 5  # Maximum length of reply chain to include
MAX_CACHED_MESSAGES_PER_CHANNEL = 1000  # Maximum number of messages to cache per channel
CHANNEL_PROCESSING_WINDOW = 1000  # Number of messages to process at once per channel
CONTEXT_WINDOW_OVERLAP = MAX_PRECEDING_MESSAGES  # How many messages to overlap between windows

# Configure paths and logging
REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]
PERSIST_DIR = os.path.join(REPO_ROOT, "vector_index")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set sentence_transformers logger to WARNING to disable progress bars
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class VectorStore:
    """
    Manages vector embeddings for semantic search using HuggingFace and Qdrant
    """
    
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, force_recreate=False):
        """Initialize the vector store with direct access to embeddings and storage"""
        # Ensure persist directory exists
        pathlib.Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Using vector index path: {PERSIST_DIR}")
        
        # Create embedding model
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=os.path.join(PERSIST_DIR, "embedding_model")
        )
        
        # Create reranker model
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        # Create qdrant client and collections
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Handle recreation if needed
        if force_recreate:
            self._delete_collections()
            
        # Ensure collections exist
        self._ensure_collections_exist()
    
    def _delete_collections(self):
        """Delete the existing collections to handle dimension changes"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            for collection_name in [COLLECTION_NAME, CONTEXT_COLLECTION_NAME]:
                if collection_name in collection_names:
                    logger.info(f"Deleting collection {collection_name}")
                    self.qdrant_client.delete_collection(collection_name=collection_name)
                    logger.info(f"Collection {collection_name} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collections: {e}")
            raise
    
    def _ensure_collections_exist(self):
        """Ensure the vector collections exist with proper configuration"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            for collection_name in [COLLECTION_NAME, CONTEXT_COLLECTION_NAME]:
                if collection_name not in collection_names:
                    logger.info(f"Creating collection {collection_name} with dimension {VECTOR_DIMENSION}")
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=VECTOR_DIMENSION,
                            distance=models.Distance.COSINE,
                        ),
                    )
                else:
                    # Verify the dimension is correct
                    collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
                    actual_dim = collection_info.config.params.vectors.size
                    if actual_dim != VECTOR_DIMENSION:
                        logger.error(f"Dimension mismatch in {collection_name}: expected {VECTOR_DIMENSION}, got {actual_dim}")
                        raise ValueError(f"Vector dimension mismatch in {collection_name}: expected {VECTOR_DIMENSION}, got {actual_dim}")
                    logger.info(f"Using existing collection {collection_name} with correct dimension {VECTOR_DIMENSION}")
        except Exception as e:
            logger.error(f"Error ensuring collections exist: {e}")
            raise
    
    def _compute_hash(self, text: str, metadata: Dict[str, Any]) -> str:
        """Compute a deterministic hash for a document"""
        # Create a string combining text and relevant metadata
        content_to_hash = text + json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
    
    def _get_message_content(self, message: Dict[str, Any]) -> str:
        """Extract content from a message including embeds"""
        content_parts = []
        
        # Get main message content
        if "content" in message:
            if isinstance(message["content"], list):
                content_items = [item.get("content", "") for item in message["content"] if "content" in item]
                if content_items:
                    content_parts.append(" ".join(content_items))
            elif isinstance(message["content"], str):
                content_parts.append(message["content"])
        
        # Get embed content
        if "embeds" in message and message["embeds"]:
            for embed in message["embeds"]:
                embed_parts = []
                
                # Add embed title if present
                if embed.get("title"):
                    embed_parts.append(f"Title: {embed['title']}")
                
                # Add embed description if present
                if embed.get("description"):
                    embed_parts.append(f"Description: {embed['description']}")
                
                # Add embed fields if present
                if embed.get("fields"):
                    for field in embed["fields"]:
                        embed_parts.append(f"{field['name']}: {field['value']}")
                
                # Add embed footer if present
                if embed.get("footer", {}).get("text"):
                    embed_parts.append(f"Footer: {embed['footer']['text']}")
                
                if embed_parts:
                    content_parts.append("Embed: " + " | ".join(embed_parts))
        
        return " | ".join(content_parts) if content_parts else ""
    
    def _get_preceding_messages(self, collection_messages, message: Dict[str, Any], max_messages: int = MAX_PRECEDING_MESSAGES) -> List[Dict[str, Any]]:
        """Get preceding messages in the same channel"""
        try:
            # Get messages before this one in the same channel
            preceding_messages = list(collection_messages.find({
                "channelId": message["channelId"],
                "_id": {"$lt": message["_id"]}
            }).sort("_id", -1).limit(max_messages))
            
            # Reverse to get chronological order
            preceding_messages.reverse()
            return preceding_messages
        except Exception as e:
            logger.error(f"Error getting preceding messages: {e}")
            return []

    def _fetch_message_batch(self, collection_messages, channel_id: str, before_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch a batch of messages before a given message ID"""
        try:
            messages = list(collection_messages.find({
                "channelId": channel_id,
                "_id": {"$lt": before_id}
            }).sort("_id", -1).limit(limit))
            messages.reverse()  # Convert to chronological order
            return messages
        except Exception as e:
            logger.error(f"Error fetching message batch: {e}")
            return []

    def _get_preceding_messages_optimized(self, collection_messages, message: Dict[str, Any], message_cache: Dict[str, List[Dict[str, Any]]], max_messages: int = MAX_PRECEDING_MESSAGES) -> List[Dict[str, Any]]:
        """Get preceding messages using a message cache to minimize database queries"""
        try:
            channel_id = message["channelId"]
            message_id = message["_id"]
            
            # Initialize cache for this channel if not exists
            if channel_id not in message_cache:
                message_cache[channel_id] = []
            
            cached_messages = message_cache[channel_id]
            
            # If we have cached messages, check if they're sufficient
            if cached_messages:
                # Find messages that are before our target message
                preceding = [msg for msg in cached_messages if msg["_id"] < message_id]
                if preceding:
                    # If we have enough preceding messages, return them
                    if len(preceding) >= max_messages:
                        return preceding[-max_messages:]
                    # If we don't have enough, we'll need to fetch more
            
            # Fetch a new batch of messages
            batch = self._fetch_message_batch(collection_messages, channel_id, message_id)
            if not batch:
                return []
            
            # Update cache with new messages
            # Merge, filter out newer messages, and limit cache size
            merged_messages = cached_messages + batch
            filtered_messages = [msg for msg in merged_messages if msg["_id"] < message_id]
            
            # Sort by _id (oldest first) and limit cache size
            filtered_messages.sort(key=lambda x: x["_id"])
            if len(filtered_messages) > MAX_CACHED_MESSAGES_PER_CHANNEL:
                # Keep the most recent messages up to the limit
                filtered_messages = filtered_messages[-MAX_CACHED_MESSAGES_PER_CHANNEL:]
            
            message_cache[channel_id] = filtered_messages
            
            # Return the messages we need
            preceding_messages = [msg for msg in filtered_messages if msg["_id"] < message_id]
            return preceding_messages[-max_messages:] if preceding_messages else []
            
        except Exception as e:
            logger.error(f"Error getting preceding messages (optimized): {e}")
            return []
    
    def _get_reply_chain(self, collection_messages, message: Dict[str, Any], max_length: int = MAX_REPLY_CHAIN_LENGTH) -> List[Dict[str, Any]]:
        """Get the chain of messages that this message is replying to"""
        try:
            reply_chain = []
            current_message = message
            
            while len(reply_chain) < max_length and current_message.get("reference"):
                # Get the referenced message
                ref_message = collection_messages.find_one({"_id": current_message["reference"]["messageId"]})
                if not ref_message:
                    break
                    
                reply_chain.append(ref_message)
                current_message = ref_message
            
            return reply_chain
        except Exception as e:
            logger.error(f"Error getting reply chain: {e}")
            return []
    
    def _build_context_text(self, message: Dict[str, Any], preceding_messages: List[Dict[str, Any]], reply_chain: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Build the context text and metadata for a message"""
        # Start with the main message content
        context_parts = []
        metadata = {
            "message_id": message["_id"],
            "channel_id": message.get("channelId", ""),
            "guild_id": message.get("guildId", ""),
            "author_name": message["author"]["name"],
            "timestamp": message.get("timestamp", ""),
            "has_reply_chain": bool(reply_chain),
            "has_preceding_messages": bool(preceding_messages),
            "has_embeds": bool(message.get("embeds"))
        }
        
        # Add reply chain if present
        if reply_chain:
            for ref_msg in reply_chain:
                ref_content = self._get_message_content(ref_msg)
                if ref_content:
                    context_parts.append(f"{ref_msg['author']['name']}: {ref_content}")
        
        # Add preceding messages if present
        if preceding_messages:
            for prev_msg in preceding_messages:
                prev_content = self._get_message_content(prev_msg)
                if prev_content:
                    context_parts.append(f"{prev_msg['author']['name']}: {prev_content}")
        
        # Add the main message
        main_content = self._get_message_content(message)
        if main_content:
            context_parts.append(f"{message['author']['name']}: {main_content}")
        
        # Join all parts with clear separators
        context_text = "\n\n".join(context_parts)
        return context_text, metadata
    
    def index_guild_messages(self, guild_id: str, max_messages: Optional[int] = None):
        """
        Index all messages for a guild using direct embedding and storage
        
        Args:
            guild_id: The guild ID to index
            max_messages: Maximum number of messages to index (None = all)
        """
        try:
            start_time = time.time()
            collection_messages = Database.get_guild_collection(guild_id, "messages")
            total_messages = collection_messages.count_documents({})
            
            if max_messages is not None:
                total_messages = min(total_messages, max_messages)
            
            logger.info(f"Starting indexing for guild {guild_id} with {total_messages} messages")
            
            # Step 1: Fetch all channels in the guild to process each channel separately
            channels = collection_messages.distinct("channelId")
            logger.info(f"Found {len(channels)} channels to process")
            
            # Track overall stats
            processed_count = 0
            valid_docs_count = 0
            indexed_count = 0
            
            # Process messages in batches
            current_batch_texts = []
            current_batch_ids = []
            current_batch_metadata = []
            current_batch_hashes = []
            
            batch_size = DEFAULT_BATCH_SIZE
            
            # Get channel names map if possible
            channel_names = {}
            try:
                channels_collection = Database.get_guild_collection(guild_id, "channels")
                for channel_doc in channels_collection.find({"_id": {"$in": channels}}):
                    channel_names[channel_doc["_id"]] = channel_doc.get("name", "unknown")
            except Exception as e:
                logger.warning(f"Could not fetch channel names: {e}")
            
            # Calculate total number of messages to process
            total_to_process = 0
            channel_message_counts = {}
            for channel_id in channels:
                count = collection_messages.count_documents({"channelId": channel_id})
                channel_message_counts[channel_id] = count
                total_to_process += count
            
            # Create overall progress bar
            overall_progress = tqdm.tqdm(total=total_to_process, desc="Processing messages", unit="msg")
            
            # Process each channel
            for channel_idx, channel_id in enumerate(channels):
                # Get channel name for display
                channel_name = channel_names.get(channel_id, f"Channel-{channel_id[:8]}")
                message_count = channel_message_counts[channel_id]
                
                logger.info(f"Processing channel {channel_idx+1}/{len(channels)}: {channel_name} ({message_count} messages)")
                
                if message_count == 0:
                    continue
                
                # Process the channel in windows to avoid loading everything into memory
                message_windows = []
                
                # Calculate number of windows needed
                if message_count <= CHANNEL_PROCESSING_WINDOW:
                    # Small enough to process in one window
                    message_windows = [(0, message_count)]
                else:
                    # Need multiple windows with overlap
                    window_size = CHANNEL_PROCESSING_WINDOW
                    effective_window = window_size - CONTEXT_WINDOW_OVERLAP
                    
                    # Create windows with overlap
                    start = 0
                    while start < message_count:
                        end = min(start + window_size, message_count)
                        message_windows.append((start, end))
                        start += effective_window
                        
                        # If the next window would be smaller than the overlap, just extend this one
                        if start < message_count and start + window_size >= message_count:
                            # Adjust the current window to include the remaining messages
                            message_windows[-1] = (message_windows[-1][0], message_count)
                            break
                
                if len(message_windows) > 1:
                    logger.info(f"Channel {channel_name} will be processed in {len(message_windows)} windows")
                
                # For each window of messages
                for window_idx, (window_start, window_end) in enumerate(message_windows):
                    # Get all messages for this window, sorted chronologically
                    window_messages = list(collection_messages.find(
                        {"channelId": channel_id}
                    ).sort("_id", 1).skip(window_start).limit(window_end - window_start))
                    
                    # Process each message in the window
                    for idx in range(len(window_messages)):
                        message = window_messages[idx]
                        processed_count += 1
                        
                        # Update overall progress
                        overall_progress.update(1)
                        
                        # Skip empty messages
                        if not self._get_message_content(message):
                            continue
                        
                        # Calculate the actual position in the channel
                        actual_idx = window_start + idx
                        
                        # For the first window or messages that are far enough from the start of a window,
                        # we can get preceding messages directly from the window
                        if window_idx == 0 or idx >= MAX_PRECEDING_MESSAGES:
                            # Get preceding messages from the current window
                            start_idx = max(0, idx - MAX_PRECEDING_MESSAGES)
                            preceding_messages = window_messages[start_idx:idx]
                        else:
                            # We're near the start of a window (but not the first one)
                            # Need to fetch preceding messages that might be in the previous window
                            preceding_messages = list(collection_messages.find({
                                "channelId": channel_id,
                                "_id": {"$lt": message["_id"]}
                            }).sort("_id", -1).limit(MAX_PRECEDING_MESSAGES))
                            preceding_messages.reverse()  # Chronological order
                        
                        # Get reply chain
                        reply_chain = self._get_reply_chain(collection_messages, message)
                        
                        # Build context text and metadata
                        context_text, metadata = self._build_context_text(message, preceding_messages, reply_chain)
                        
                        # Create a deterministic document ID
                        message_id = message["_id"]
                        combined_id = f"{guild_id}_{message_id}"
                        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined_id))
                        
                        # Compute hash for deduplication and updates
                        content_hash = self._compute_hash(context_text, metadata)
                        
                        valid_docs_count += 1
                        
                        # Add to current batch
                        current_batch_texts.append(context_text)
                        current_batch_ids.append(doc_id)
                        current_batch_metadata.append(metadata)
                        current_batch_hashes.append(content_hash)
                        
                        # When batch is full, process it
                        if len(current_batch_texts) >= batch_size:
                            batch_indexed = self._process_batch(
                                current_batch_texts, 
                                current_batch_ids, 
                                current_batch_metadata,
                                current_batch_hashes,
                                CONTEXT_COLLECTION_NAME
                            )
                            indexed_count += batch_indexed
                            
                            # Clear batches for next round
                            current_batch_texts = []
                            current_batch_ids = []
                            current_batch_metadata = []
                            current_batch_hashes = []
            
            # Process any remaining documents
            if current_batch_texts:
                batch_indexed = self._process_batch(
                    current_batch_texts, 
                    current_batch_ids, 
                    current_batch_metadata,
                    current_batch_hashes,
                    CONTEXT_COLLECTION_NAME
                )
                indexed_count += batch_indexed
            
            overall_progress.close()
            
            elapsed_time = time.time() - start_time
            avg_time_per_message = elapsed_time / processed_count if processed_count > 0 else 0
            logger.info(f"Indexing complete:")
            logger.info(f"- Total messages processed: {processed_count}")
            logger.info(f"- Total documents indexed: {indexed_count}")
            logger.info(f"- Total time: {elapsed_time:.2f} seconds")
            logger.info(f"- Average time per message: {avg_time_per_message:.4f} seconds")
            
            return {"indexed_messages": indexed_count, "processed_messages": processed_count, "elapsed_time": elapsed_time}
        
        except Exception as e:
            logger.error(f"Error indexing guild messages: {e}")
            raise
    
    def _process_batch(self, texts, ids, metadata_list, hashes, collection_name: str):
        """
        Process a batch of documents efficiently
        
        Returns the number of documents that were added or updated
        """
        try:
            # Get existing points
            existing_points = self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False
            )
            
            # Organize existing points by ID
            existing_dict = {str(point.id): point for point in existing_points}
            
            # Determine which points need to be updated or added
            points_to_upsert = []
            processed_count = 0
            
            # Generate embeddings for all texts in one batch
            # print("Generating embeddings for", json.dumps(texts, indent=2))
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            for i, doc_id in enumerate(ids):
                # Check if document exists and if hash matches
                if doc_id in existing_dict:
                    existing_hash = existing_dict[doc_id].payload.get("hash")
                    if existing_hash == hashes[i]:
                        # Document exists and hasn't changed, skip it
                        continue
                
                # Document is new or has changed, add to upsert list
                points_to_upsert.append(
                    PointStruct(
                        id=doc_id,
                        vector=embeddings[i].tolist(),
                        payload={
                            **metadata_list[i],
                            "text": texts[i],
                            "hash": hashes[i]
                        }
                    )
                )
                processed_count += 1
            
            # Perform batch upsert if there are any points to update
            if points_to_upsert:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points_to_upsert
                )
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def delete_guild_documents(self, guild_id: str):
        """Delete all documents for a specific guild"""
        try:
            # Create a filter for the guild ID
            filter_param = Filter(
                must=[
                    FieldCondition(
                        key="guild_id",
                        match=MatchValue(value=guild_id)
                    )
                ]
            )
            
            total_deleted = 0
            
            # Delete from both collections
            for collection_name in [COLLECTION_NAME, CONTEXT_COLLECTION_NAME]:
                # Count documents to be deleted
                count_response = self.qdrant_client.count(
                    collection_name=collection_name,
                    count_filter=filter_param
                )
                count = count_response.count
                
                if count > 0:
                    logger.info(f"Deleting {count} documents for guild {guild_id} from {collection_name}")
                    
                    # Delete documents matching the filter
                    self.qdrant_client.delete(
                        collection_name=collection_name,
                        points_selector=filter_param
                    )
                    
                    total_deleted += count
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error deleting guild documents: {e}")
            raise
    
    def search(self, query: str, limit: int = 10, similarity_cutoff: float = 0.5, use_context: bool = True) -> List[Dict[str, Any]]:
        """
        Search for messages using semantic search with reranking
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            similarity_cutoff: Minimum similarity score threshold (0-1)
            use_context: Whether to search in the context-aware collection
            
        Returns:
            List of messages matching the query, reranked by relevance
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(BGE_QUERY_PREFIX + query, show_progress_bar=False)
            
            # Get more results than needed for reranking
            initial_limit = min(limit * 2, 50)  # Get up to 2x the requested limit, max 50
            
            # Choose which collection to search
            collection_name = CONTEXT_COLLECTION_NAME if use_context else COLLECTION_NAME
            
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=initial_limit,
                score_threshold=similarity_cutoff
            )
            
            if not search_results:
                return []
            
            # Prepare pairs for reranking
            pairs = []
            for result in search_results:
                pairs.append((query, result.payload.get("text", "")))
            
            # Rerank the results
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine results with rerank scores
            reranked_results = []
            for result, rerank_score in zip(search_results, rerank_scores):
                payload = result.payload
                reranked_results.append({
                    "message_id": payload.get("message_id"),
                    "channel_id": payload.get("channel_id"),
                    "guild_id": payload.get("guild_id"),
                    "content": payload.get("text"),
                    "author_name": payload.get("author_name"),
                    "timestamp": payload.get("timestamp"),
                    "point_id": str(result.id),
                    "vector_score": result.score,
                    "rerank_score": float(rerank_score),
                    "has_reply_chain": payload.get("has_reply_chain", False),
                    "has_preceding_messages": payload.get("has_preceding_messages", False)
                })
            
            # Sort by rerank score and take top results
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def count_matches(self, query: str, guild_id: str = None, similarity_cutoff: float = 0.7, use_context: bool = True) -> int:
        """
        Count the number of messages matching a semantic search query without retrieving full results
        
        Args:
            query: The search query
            guild_id: Optional guild ID to filter results by
            similarity_cutoff: Minimum similarity score threshold (0-1)
            use_context: Whether to search in the context-aware collection
            
        Returns:
            Count of matching messages
        """
        try:
            # Prepare filter if guild_id is provided
            search_filter = None
            if guild_id:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="guild_id",
                            match=MatchValue(value=guild_id)
                        )
                    ]
                )
            
            # For semantic search with similarity threshold, we need to use search
            # Generate query embedding
            query_embedding = self.embedding_model.encode(BGE_QUERY_PREFIX + query, show_progress_bar=False)
            
            # Choose which collection to search
            collection_name = CONTEXT_COLLECTION_NAME if use_context else COLLECTION_NAME
            
            # Use search with high limit to get approximate count
            # This is necessary because Qdrant doesn't support vector similarity in count queries
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=1000,  # High limit to get a good estimate
                score_threshold=similarity_cutoff,
                query_filter=search_filter
            )
            
            return len(search_results)
            
        except Exception as e:
            logger.error(f"Error counting matches: {e}")
            raise
