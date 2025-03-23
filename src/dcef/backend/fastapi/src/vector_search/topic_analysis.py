import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, Path
import os
import numpy as np
from sklearn.preprocessing import normalize

from .vector_store import VectorStore, VECTOR_DIMENSION
from .topic_clustering import TopicClusterer
from ..common.Database import Database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="",
    tags=["topic_analysis"]
)

# Cache instances
_topic_clusterer = None
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get or create VectorStore instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def get_topic_clusterer() -> TopicClusterer:
    """Get or create TopicClusterer instance"""
    global _topic_clusterer
    if _topic_clusterer is None:
        vector_store = get_vector_store()
        _topic_clusterer = TopicClusterer(vector_store)
    return _topic_clusterer

@router.get("/channel/analyze_topics")
async def analyze_channel_topics(
    guild: str,
    channel: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_cluster_size: int = Query(5, ge=3, le=50),
    output_dir: str = Query("./topic_analysis")
) -> Dict[str, Any]:
    """
    Analyze and visualize topics in a Discord channel
    
    Args:
        guild: Discord guild ID
        channel: Discord channel ID
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        min_cluster_size: Minimum cluster size for topic detection
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get topic clusterer
        topic_clusterer = get_topic_clusterer()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the channel
        results = topic_clusterer.process_channel(
            guild_id=guild,
            channel_id=channel,
            start_date=start_date,
            end_date=end_date,
            min_cluster_size=min_cluster_size,
            output_dir=output_dir
        )
        
        # Extract topic summaries for the response
        topic_summaries = []
        
        if "topic_segments" in results:
            # Create a mapping from topic_id to segments
            topic_map = {}
            for segment in results["topic_segments"]:
                topic_id = segment["topic_id"]
                if topic_id not in topic_map:
                    topic_map[topic_id] = []
                topic_map[topic_id].append(segment)
            
            # Create summaries for each topic (combining segments if needed)
            for topic_id, segments in topic_map.items():
                # Skip noise cluster
                if topic_id == -1:
                    continue
                    
                # Sort segments by message count (descending)
                segments.sort(key=lambda x: x["message_count"], reverse=True)
                
                # Use the largest segment as representative
                main_segment = segments[0]
                
                # Calculate total messages across all segments of this topic
                total_messages = sum(segment["message_count"] for segment in segments)
                
                # Only include topics with significant message count
                if total_messages >= 5:
                    summary = main_segment.get("summary", {})
                    
                    # Create a clean summary
                    topic_summary = {
                        "topic_id": topic_id,
                        "total_messages": total_messages,
                        "segment_count": len(segments),
                        "representative_messages": summary.get("sample_messages", [])[:3],
                        "top_terms": summary.get("top_terms", [])[:8],
                        "top_authors": summary.get("top_authors", [])[:3],
                    }
                    
                    topic_summaries.append(topic_summary)
            
            # Sort topics by total message count
            topic_summaries.sort(key=lambda x: x["total_messages"], reverse=True)
        
        # Return results
        return {
            "status": "success",
            "guild_id": guild,
            "channel_id": channel,
            "message_count": results.get("message_count", 0),
            "topic_count": results.get("topic_count", 0),
            "visualization_files": [
                f"channel_{channel}_timeline.png",
                f"channel_{channel}_timeline_distribution.png",
                f"channel_{channel}_clusters.png",
                f"channel_{channel}_semantic_distances.png",
                f"channel_{channel}_results.json"
            ],
            "top_topics": topic_summaries[:10],  # Return top 10 topics
            "output_dir": output_dir
        }
    
    except Exception as e:
        logger.error(f"Error analyzing channel topics: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing channel topics: {str(e)}")

@router.get("/channel/analyze_semantic_boundaries")
async def analyze_semantic_boundaries(
    guild: str,
    channel: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    percentile_threshold: float = Query(95.0, ge=80.0, le=99.9),
    output_dir: str = Query("./topic_analysis")
) -> Dict[str, Any]:
    """
    Analyze semantic boundaries in a Discord channel based on message embeddings
    
    Args:
        guild: Discord guild ID
        channel: Discord channel ID
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        percentile_threshold: Percentile threshold for determining significant distance changes
                             (higher = fewer boundaries, lower = more boundaries)
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get topic clusterer
        topic_clusterer = get_topic_clusterer()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Fetch messages
        messages = topic_clusterer.fetch_channel_messages(
            guild_id=guild,
            channel_id=channel,
            start_date=start_date,
            end_date=end_date
        )
        
        if not messages:
            return {
                "status": "success",
                "message": "No messages found in the specified range",
                "message_count": 0
            }
        
        # Step 2: Identify semantic boundaries
        semantic_segments = topic_clusterer.identify_semantic_boundaries(
            messages=messages,
            percentile_threshold=percentile_threshold
        )
        
        # Step 3: Generate visualization
        output_base = f"{output_dir}/channel_{channel}"
        visualizations = []
        
        try:
            # Save semantic distances visualization
            distance_plot_path = f"{output_base}_semantic_distances.png"
            topic_clusterer.visualize_semantic_distances(
                messages=messages,
                topic_segments=semantic_segments,
                output_path=distance_plot_path
            )
            visualizations.append(f"channel_{channel}_semantic_distances.png")
        except Exception as e:
            logger.warning(f"Error generating semantic distances visualization: {e}")
        
        try:
            # Generate timeline visualization for semantic segments
            timeline_path = f"{output_base}_semantic_timeline.png"
            topic_clusterer.visualize_semantic_timeline(
                semantic_segments, 
                timeline_path
            )
            visualizations.append(f"channel_{channel}_semantic_timeline.png")
            visualizations.append(f"channel_{channel}_semantic_timeline_counts.png")
        except Exception as e:
            logger.warning(f"Error generating semantic timeline visualization: {e}")
        
        # Extract boundary information for the response
        boundaries = []
        for i, segment in enumerate(semantic_segments):
            if i == 0:  # First segment has no boundary
                continue
                
            boundary = {
                "segment_id": i,
                "timestamp": segment.get("start_timestamp"),
                "message_count_before": semantic_segments[i-1].get("message_count", 0),
                "message_count_after": segment.get("message_count", 0),
                "distance": segment.get("boundary_distance", 0)
            }
            boundaries.append(boundary)
            
        # Sort boundaries by distance (descending)
        boundaries.sort(key=lambda x: x.get("distance", 0), reverse=True)
        
        # Return results
        return {
            "status": "success",
            "guild_id": guild,
            "channel_id": channel,
            "message_count": len(messages),
            "segment_count": len(semantic_segments),
            "boundary_count": len(boundaries),
            "percentile_threshold": percentile_threshold,
            "visualization_files": visualizations,
            "top_boundaries": boundaries[:10],  # Return top 10 boundaries by distance
            "output_dir": output_dir
        }
    
    except Exception as e:
        logger.error(f"Error analyzing semantic boundaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing semantic boundaries: {str(e)}")

@router.get("/test/topic_clustering")
async def test_topic_clustering() -> Dict[str, Any]:
    """
    Test the topic clustering functionality with a simple example
    
    Returns:
        Dictionary with test results
    """
    try:
        # Get topic clusterer
        topic_clusterer = get_topic_clusterer()
        
        # Log test start
        logger.info("Starting topic clustering test")
        
        # Check vector store connection
        vector_store = get_vector_store()
        qdrant_client = vector_store.qdrant_client
        
        # Get collection info to verify connection
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Return basic status info
        return {
            "status": "success",
            "message": "Topic clustering module loaded successfully",
            "vector_collections": collection_names,
            "module_info": {
                "clusterer": str(topic_clusterer.__class__.__name__),
                "vector_store": str(vector_store.__class__.__name__)
            }
        }
    
    except Exception as e:
        logger.error(f"Error testing topic clustering: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing topic clustering: {str(e)}")

@router.get("/test/semantic_boundaries")
async def test_semantic_boundaries() -> Dict[str, Any]:
    """
    Test the semantic boundary detection with synthetic data
    
    Returns:
        Dictionary with test results
    """
    try:
        # Get topic clusterer
        topic_clusterer = get_topic_clusterer()
        
        # Create synthetic data with known topic shifts
        synthetic_messages = []
        
        # Create synthetic vectors for different topics
        np.random.seed(42)  # For reproducibility
        
        # Topic 1: Technology
        tech_vectors = normalize(np.random.normal(size=(10, VECTOR_DIMENSION)))
        for i in range(10):
            timestamp = f"2023-03-01T{10+i:02d}:00:00Z"
            synthetic_messages.append({
                "id": f"tech_{i}",
                "vector": tech_vectors[i],
                "content": f"Technology message {i}",
                "author_name": "TechUser",
                "timestamp": timestamp
            })
        
        # Topic 2: Sports (should have higher distance from Technology)
        sports_vectors = normalize(np.random.normal(size=(10, VECTOR_DIMENSION)))
        for i in range(10):
            timestamp = f"2023-03-01T{20+i:02d}:00:00Z"
            synthetic_messages.append({
                "id": f"sports_{i}",
                "vector": sports_vectors[i],
                "content": f"Sports message {i}",
                "author_name": "SportsUser",
                "timestamp": timestamp
            })
        
        # Topic 3: Music
        music_vectors = normalize(np.random.normal(size=(10, VECTOR_DIMENSION)))
        for i in range(10):
            timestamp = f"2023-03-02T{10+i:02d}:00:00Z"
            synthetic_messages.append({
                "id": f"music_{i}",
                "vector": music_vectors[i],
                "content": f"Music message {i}",
                "author_name": "MusicUser",
                "timestamp": timestamp
            })
        
        # Identify semantic boundaries
        semantic_segments = topic_clusterer.identify_semantic_boundaries(
            messages=synthetic_messages,
            percentile_threshold=90.0  # Lower threshold to ensure we get boundaries
        )
        
        # Generate visualizations
        output_dir = "./topic_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = []
        
        try:
            # Save semantic distances visualization
            distance_path = f"{output_dir}/test_semantic_distances.png"
            topic_clusterer.visualize_semantic_distances(
                messages=synthetic_messages,
                topic_segments=semantic_segments,
                output_path=distance_path
            )
            visualizations.append("test_semantic_distances.png")
        except Exception as e:
            logger.warning(f"Error generating test semantic distances visualization: {e}")
        
        try:
            # Generate timeline visualization for semantic segments
            timeline_path = f"{output_dir}/test_semantic_timeline.png"
            topic_clusterer.visualize_semantic_timeline(
                semantic_segments, 
                timeline_path
            )
            visualizations.append("test_semantic_timeline.png")
            visualizations.append("test_semantic_timeline_counts.png")
        except Exception as e:
            logger.warning(f"Error generating test semantic timeline visualization: {e}")
        
        # Return information about the segments
        segment_info = []
        for i, segment in enumerate(semantic_segments):
            segment_info.append({
                "segment_id": i,
                "message_count": segment.get("message_count", 0),
                "start_time": segment.get("start_timestamp", ""),
                "end_time": segment.get("end_timestamp", ""),
                "boundary_distance": segment.get("boundary_distance", 0) if i > 0 else None
            })
        
        return {
            "status": "success",
            "message": "Semantic boundary detection test completed",
            "total_messages": len(synthetic_messages),
            "segment_count": len(semantic_segments),
            "segments": segment_info,
            "visualization_files": visualizations
        }
    
    except Exception as e:
        logger.error(f"Error testing semantic boundaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing semantic boundaries: {str(e)}")

@router.get("/channel/topic_stats")
async def get_channel_topic_stats(
    guild: str,
    channel: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get basic statistics about the messages in a channel without full clustering
    
    Args:
        guild: Discord guild ID
        channel: Discord channel ID
        start_date: Optional start date in ISO format (YYYY-MM-DD)
        end_date: Optional end date in ISO format (YYYY-MM-DD)
        
    Returns:
        Dictionary with channel statistics
    """
    try:
        # Check database connection
        if not Database.is_online():
            raise HTTPException(status_code=503, detail="Database is not available")
        
        # Get topic clusterer
        topic_clusterer = get_topic_clusterer()
        
        # Fetch messages
        messages = topic_clusterer.fetch_channel_messages(
            guild_id=guild,
            channel_id=channel,
            start_date=start_date,
            end_date=end_date
        )
        
        if not messages:
            return {
                "status": "success",
                "message_count": 0,
                "channel_id": channel,
                "guild_id": guild
            }
        
        # Compute basic statistics
        authors = {}
        min_timestamp = None
        max_timestamp = None
        
        for msg in messages:
            # Track authors
            author = msg.get("author_name")
            if author:
                authors[author] = authors.get(author, 0) + 1
                
            # Track timestamps
            timestamp = msg.get("timestamp")
            if timestamp:
                if min_timestamp is None or timestamp < min_timestamp:
                    min_timestamp = timestamp
                if max_timestamp is None or timestamp > max_timestamp:
                    max_timestamp = timestamp
        
        # Return statistics
        return {
            "status": "success",
            "channel_id": channel,
            "guild_id": guild,
            "message_count": len(messages),
            "author_count": len(authors),
            "top_authors": sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10],
            "date_range": {
                "start": min_timestamp,
                "end": max_timestamp
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting channel topic stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting channel topic stats: {str(e)}")
