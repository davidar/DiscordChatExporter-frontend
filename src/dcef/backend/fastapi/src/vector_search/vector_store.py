import os
import logging
from typing import List, Dict, Any, Optional
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
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Optimized for reranking
VECTOR_DIMENSION = 384  # BGE-small-en-v1.5 dimensions
DEFAULT_BATCH_SIZE = 1024
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

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
        
        # Create qdrant client and collection
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Handle recreation if needed
        if force_recreate:
            self._delete_collection()
            
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _delete_collection(self):
        """Delete the existing collection to handle dimension changes"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if COLLECTION_NAME in collection_names:
                logger.info(f"Deleting collection {COLLECTION_NAME}")
                self.qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                logger.info(f"Collection {COLLECTION_NAME} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the vector collection exists with proper configuration"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection {COLLECTION_NAME} with dimension {VECTOR_DIMENSION}")
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=VECTOR_DIMENSION,
                        distance=models.Distance.COSINE,
                    ),
                )
            else:
                # Verify the dimension is correct
                collection_info = self.qdrant_client.get_collection(collection_name=COLLECTION_NAME)
                actual_dim = collection_info.config.params.vectors.size
                if actual_dim != VECTOR_DIMENSION:
                    logger.error(f"Dimension mismatch: expected {VECTOR_DIMENSION}, got {actual_dim}")
                    raise ValueError(f"Vector dimension mismatch: expected {VECTOR_DIMENSION}, got {actual_dim}")
                logger.info(f"Using existing collection {COLLECTION_NAME} with correct dimension {VECTOR_DIMENSION}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def _compute_hash(self, text: str, metadata: Dict[str, Any]) -> str:
        """Compute a deterministic hash for a document"""
        # Create a string combining text and relevant metadata
        content_to_hash = text + json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
    
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
            
            # Get cursor for messages
            cursor = collection_messages.find({})
            if max_messages is not None:
                cursor = cursor.limit(max_messages)
            
            # Setup progress tracking
            progress_bar = tqdm.tqdm(cursor, total=total_messages, desc="Processing", unit="msg")
            processed_count = 0
            valid_docs_count = 0
            indexed_count = 0
            
            # Process messages in batches
            current_batch_texts = []
            current_batch_ids = []
            current_batch_metadata = []
            current_batch_hashes = []
            
            batch_size = DEFAULT_BATCH_SIZE
            
            # Process all messages
            for message in progress_bar:
                processed_count += 1
                
                # Extract the content
                message_content = ""
                if "content" in message:
                    if isinstance(message["content"], list):
                        content_items = [item.get("content", "") for item in message["content"] if "content" in item]
                        message_content = " ".join(content_items)
                    elif isinstance(message["content"], str):
                        message_content = message["content"]
                
                # Skip empty messages
                if not message_content:
                    continue
                
                # Extract the message ID and create a deterministic document ID
                message_id = message["_id"]
                combined_id = f"{guild_id}_{message_id}"
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, combined_id))
                
                channel_id = message.get("channelId", "")
                author_name = message.get("author", {}).get("username", "")
                if not author_name and "author" in message and "name" in message["author"]:
                    author_name = message["author"]["name"]
                timestamp = message.get("timestamp", "")
                
                # Create metadata
                metadata = {
                    "message_id": message_id,
                    "channel_id": channel_id,
                    "guild_id": guild_id,
                    "author_name": author_name,
                    "timestamp": timestamp
                }
                
                # Compute hash for deduplication and updates
                content_hash = self._compute_hash(message_content, metadata)
                
                valid_docs_count += 1
                
                # Add to current batch
                current_batch_texts.append(message_content)
                current_batch_ids.append(doc_id)
                current_batch_metadata.append(metadata)
                current_batch_hashes.append(content_hash)
                
                # When batch is full, process it
                if len(current_batch_texts) >= batch_size:
                    batch_indexed = self._process_batch(
                        current_batch_texts, 
                        current_batch_ids, 
                        current_batch_metadata,
                        current_batch_hashes
                    )
                    indexed_count += batch_indexed
                    
                    # Clear batches for next round
                    current_batch_texts = []
                    current_batch_ids = []
                    current_batch_metadata = []
                    current_batch_hashes = []
                
                # Update progress
                # progress_bar.set_postfix(valid=valid_docs_count, indexed=indexed_count, batch=len(current_batch_texts))
            
            # Process any remaining documents
            if current_batch_texts:
                batch_indexed = self._process_batch(
                    current_batch_texts, 
                    current_batch_ids, 
                    current_batch_metadata,
                    current_batch_hashes
                )
                indexed_count += batch_indexed
            
            progress_bar.close()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Indexed {indexed_count} documents, processed {processed_count} messages in {elapsed_time:.2f} seconds")
            
            return {"indexed_messages": indexed_count, "processed_messages": processed_count, "elapsed_time": elapsed_time}
        
        except Exception as e:
            logger.error(f"Error indexing guild messages: {e}")
            raise
    
    def _process_batch(self, texts, ids, metadata_list, hashes):
        """
        Process a batch of documents efficiently
        
        Returns the number of documents that were added or updated
        """
        # Check which points already exist and their current hashes
        try:
            # Get existing points
            existing_points = self.qdrant_client.retrieve(
                collection_name=COLLECTION_NAME,
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
                    collection_name=COLLECTION_NAME,
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
            
            # Count documents to be deleted
            count_response = self.qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=filter_param
            )
            count = count_response.count
            
            if count > 0:
                logger.info(f"Deleting {count} documents for guild {guild_id}")
                
                # Delete documents matching the filter
                self.qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=filter_param
                )
                
                return count
            
            return 0
        except Exception as e:
            logger.error(f"Error deleting guild documents: {e}")
            raise
    
    def search(self, query: str, limit: int = 10, similarity_cutoff: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for messages using semantic search with reranking
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            similarity_cutoff: Minimum similarity score threshold (0-1)
            
        Returns:
            List of messages matching the query, reranked by relevance
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(BGE_QUERY_PREFIX + query, show_progress_bar=False)
            
            # Get more results than needed for reranking
            initial_limit = min(limit * 2, 50)  # Get up to 2x the requested limit, max 50
            search_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
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
                    "rerank_score": float(rerank_score)
                })
            
            # Sort by rerank score and take top results
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def count_matches(self, query: str, guild_id: str = None, similarity_cutoff: float = 0.7) -> int:
        """
        Count the number of messages matching a semantic search query without retrieving full results
        
        Args:
            query: The search query
            guild_id: Optional guild ID to filter results by
            similarity_cutoff: Minimum similarity score threshold (0-1)
            
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
            
            # Use search with high limit to get approximate count
            # This is necessary because Qdrant doesn't support vector similarity in count queries
            search_results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=1000,  # High limit to get a good estimate
                score_threshold=similarity_cutoff,
                query_filter=search_filter
            )
            
            return len(search_results)
            
        except Exception as e:
            logger.error(f"Error counting matches: {e}")
            raise
