import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
import uuid
from datetime import datetime
import pathlib
import json

# For clustering
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# For visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# For dimensionality reduction
import umap

# Local imports
from ..common.Database import Database
from .vector_store import VectorStore, COLLECTION_NAME

# Qdrant imports
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicClusterer:
    """
    Performs topic clustering on Discord messages using embeddings from the vector store
    """
    
    def __init__(self, vector_store: VectorStore):
        """Initialize with access to the vector store for embeddings"""
        self.vector_store = vector_store
        self.qdrant_client = vector_store.qdrant_client
    
    def fetch_channel_messages(self, guild_id: str, channel_id: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch messages from a specific channel within a date range
        
        Args:
            guild_id: Discord guild ID
            channel_id: Discord channel ID
            start_date: Optional start date in ISO format (YYYY-MM-DD)
            end_date: Optional end date in ISO format (YYYY-MM-DD)
            limit: Maximum number of messages to fetch
            
        Returns:
            List of message documents with embeddings
        """
        try:
            logger.info(f"Fetching messages for guild={guild_id}, channel={channel_id}")
            
            # Build conditions
            conditions = []
            conditions.append(FieldCondition(key="guild_id", match=MatchValue(value=guild_id)))
            conditions.append(FieldCondition(key="channel_id", match=MatchValue(value=channel_id)))
            
            # Add date range if specified
            if start_date or end_date:
                range_params = {}
                if start_date:
                    range_params["gte"] = start_date
                if end_date:
                    range_params["lte"] = end_date
                
                conditions.append(FieldCondition(key="timestamp", range=Range(**range_params)))
            
            # Create filter
            filter_obj = Filter(must=conditions)
            
            messages = []
            offset = None
            
            # Use scroll to paginate through results
            while True:
                # Call scroll with proper arguments
                response = self.qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=filter_obj,  # Use scroll_filter instead of filter
                    limit=1000,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset
                )
                
                # Process results
                points, next_offset = response
                
                if not points:
                    break
                
                for point in points:
                    message = {
                        "id": str(point.id),
                        "vector": np.array(point.vector),
                        "message_id": point.payload.get("message_id"),
                        "channel_id": point.payload.get("channel_id"),
                        "guild_id": point.payload.get("guild_id"),
                        "content": point.payload.get("text"),
                        "author_name": point.payload.get("author_name"),
                        "timestamp": point.payload.get("timestamp")
                    }
                    messages.append(message)
                
                if limit and len(messages) >= limit:
                    messages = messages[:limit]
                    break
                
                # Stop if there are no more pages
                if next_offset is None:
                    break
                    
                offset = next_offset
            
            # Sort messages by timestamp
            messages.sort(key=lambda x: x.get("timestamp", ""))
            
            logger.info(f"Retrieved {len(messages)} messages from channel {channel_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error fetching channel messages: {e}")
            raise
    
    def cluster_messages(self, messages: List[Dict[str, Any]], 
                       min_cluster_size: int = 5,
                       min_samples: int = 2) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Cluster messages using HDBSCAN to identify topics
        
        Args:
            messages: List of message documents with vector embeddings
            min_cluster_size: Minimum size of clusters
            min_samples: HDBSCAN min_samples parameter
            
        Returns:
            Tuple of (processed messages with cluster labels, reduced vectors for visualization, cluster labels)
        """
        try:
            if not messages:
                logger.warning("No messages provided for clustering")
                return [], np.array([]), np.array([])
            
            # Extract vectors and normalize them
            vectors = np.array([msg["vector"] for msg in messages])
            vectors_normalized = normalize(vectors)
            
            # Run HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',  # Use euclidean for normalized vectors
                core_dist_n_jobs=-1  # Use all CPU cores
            )
            
            cluster_labels = clusterer.fit_predict(vectors_normalized)
            
            # Add cluster labels to messages
            for i, msg in enumerate(messages):
                msg["cluster"] = int(cluster_labels[i])
            
            # Generate UMAP projection for visualization
            umap_reducer = umap.UMAP(
                n_neighbors=15,
                n_components=2,
                metric='euclidean',
                random_state=42
            )
            umap_embedding = umap_reducer.fit_transform(vectors_normalized)
            
            logger.info(f"Clustered {len(messages)} messages into {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} topics")
            return messages, umap_embedding, cluster_labels
            
        except Exception as e:
            logger.error(f"Error clustering messages: {e}")
            raise
    
    def identify_topic_boundaries(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify boundaries between topics in the message timeline
        
        Args:
            messages: List of message documents with cluster labels
            
        Returns:
            List of topic segments with metadata
        """
        if not messages:
            return []
        
        # Sort messages by timestamp
        messages.sort(key=lambda x: x.get("timestamp", ""))
        
        # Track current topic and boundaries
        current_cluster = None
        topic_segments = []
        current_segment = None
        
        for i, msg in enumerate(messages):
            cluster_id = msg.get("cluster", -1)
            
            # Skip noise points for boundary detection
            if cluster_id == -1:
                continue
                
            # Start a new segment if cluster changes
            if cluster_id != current_cluster:
                # Close previous segment
                if current_segment:
                    current_segment["end_idx"] = i - 1
                    current_segment["end_timestamp"] = messages[i-1].get("timestamp")
                    current_segment["message_count"] = current_segment["end_idx"] - current_segment["start_idx"] + 1
                    topic_segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    "cluster_id": cluster_id,
                    "start_idx": i,
                    "start_timestamp": msg.get("timestamp"),
                    "messages": []
                }
                current_cluster = cluster_id
            
            # Add message to current segment
            if current_segment:
                current_segment["messages"].append(msg)
        
        # Close the last segment
        if current_segment:
            current_segment["end_idx"] = len(messages) - 1
            current_segment["end_timestamp"] = messages[-1].get("timestamp")
            current_segment["message_count"] = current_segment["end_idx"] - current_segment["start_idx"] + 1
            topic_segments.append(current_segment)
        
        logger.info(f"Identified {len(topic_segments)} topic segments")
        return topic_segments

    def identify_semantic_boundaries(self, messages: List[Dict[str, Any]], 
                                    percentile_threshold: float = 95.0) -> List[Dict[str, Any]]:
        """
        Identify topic boundaries based on semantic distance between consecutive messages
        
        This implements Level 4 semantic splitting from the text splitting hierarchy,
        which looks at cosine distances between consecutive message embeddings to find
        natural break points in the conversation.
        
        Args:
            messages: List of message documents with vector embeddings
            percentile_threshold: Percentile threshold for determining significant distance
                                 (higher = fewer boundaries, lower = more boundaries)
            
        Returns:
            List of topic segments with metadata
        """
        if not messages or len(messages) < 2:
            return []
        
        # Sort messages by timestamp
        messages.sort(key=lambda x: x.get("timestamp", ""))
        
        # Extract vectors and calculate cosine distances between consecutive messages
        vectors = []
        valid_messages = []
        
        for msg in messages:
            if "vector" in msg:
                vectors.append(msg["vector"])
                valid_messages.append(msg)
        
        if len(valid_messages) < 2:
            logger.warning("Not enough valid messages with vectors for semantic boundary detection")
            return []
            
        # Normalize vectors
        vectors_normalized = normalize(np.array(vectors))
        
        # Calculate cosine similarity between consecutive messages
        cosine_similarities = []
        for i in range(len(vectors_normalized) - 1):
            sim = cosine_similarity(
                vectors_normalized[i].reshape(1, -1), 
                vectors_normalized[i+1].reshape(1, -1)
            )[0][0]
            cosine_similarities.append(sim)
        
        # Convert to distances (1 - similarity)
        distances = [1 - sim for sim in cosine_similarities]
        
        # Determine threshold based on percentile
        threshold = np.percentile(distances, percentile_threshold)
        logger.info(f"Semantic distance threshold (p{percentile_threshold}): {threshold:.4f}")
        
        # Find boundary points where distance exceeds threshold
        boundary_indices = [i for i, dist in enumerate(distances) if dist > threshold]
        
        # Create topic segments
        topic_segments = []
        start_idx = 0
        
        # Each boundary represents the end of a segment
        for boundary_idx in boundary_indices:
            # Boundary index in distances = index of the first message in the next segment
            end_idx = boundary_idx
            if end_idx - start_idx < 2:  # Skip segments with fewer than 2 messages
                continue
                
            segment = {
                "topic_id": len(topic_segments),  # Assign sequential topic IDs
                "boundary_type": "semantic",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_timestamp": valid_messages[start_idx].get("timestamp"),
                "end_timestamp": valid_messages[end_idx].get("timestamp"),
                "message_count": end_idx - start_idx + 1,
                "boundary_distance": distances[boundary_idx],
                "messages": valid_messages[start_idx:end_idx+1]
            }
            topic_segments.append(segment)
            
            # The next segment starts after this boundary
            start_idx = boundary_idx + 1
        
        # Add the final segment
        if start_idx < len(valid_messages) - 1:
            segment = {
                "topic_id": len(topic_segments),
                "boundary_type": "semantic",
                "start_idx": start_idx,
                "end_idx": len(valid_messages) - 1,
                "start_timestamp": valid_messages[start_idx].get("timestamp"),
                "end_timestamp": valid_messages[-1].get("timestamp"),
                "message_count": len(valid_messages) - start_idx,
                "messages": valid_messages[start_idx:]
            }
            topic_segments.append(segment)
        
        logger.info(f"Identified {len(topic_segments)} semantic topic segments")
        return topic_segments
                
    def visualize_semantic_distances(self, messages: List[Dict[str, Any]], 
                                    topic_segments: List[Dict[str, Any]] = None,
                                    output_path: str = "semantic_distances.png"):
        """
        Generate a visualization of semantic distances between consecutive messages
        and mark the identified topic boundaries
        
        Args:
            messages: List of message documents with vector embeddings
            topic_segments: Optional list of topic segments with boundary information
            output_path: Path to save the visualization
        """
        if not messages or len(messages) < 2:
            logger.warning("Not enough messages to visualize semantic distances")
            return
        
        # Sort messages by timestamp and extract vectors
        messages.sort(key=lambda x: x.get("timestamp", ""))
        
        vectors = []
        valid_indices = []
        timestamps = []
        
        for i, msg in enumerate(messages):
            if "vector" in msg:
                vectors.append(msg["vector"])
                valid_indices.append(i)
                timestamps.append(msg.get("timestamp", ""))
        
        if len(vectors) < 2:
            logger.warning("Not enough valid messages with vectors for visualization")
            return
            
        # Normalize vectors
        vectors_normalized = normalize(np.array(vectors))
        
        # Calculate cosine similarity between consecutive messages
        cosine_similarities = []
        for i in range(len(vectors_normalized) - 1):
            sim = cosine_similarity(
                vectors_normalized[i].reshape(1, -1), 
                vectors_normalized[i+1].reshape(1, -1)
            )[0][0]
            cosine_similarities.append(sim)
        
        # Convert to distances (1 - similarity)
        distances = [1 - sim for sim in cosine_similarities]
        
        if not distances:
            logger.warning("No distances calculated, can't create visualization")
            return
        
        # Determine thresholds for visualization
        p90 = np.percentile(distances, 90)
        p95 = np.percentile(distances, 95)
        p99 = np.percentile(distances, 99)
        
        # Convert timestamps to datetime objects for plotting
        datetime_objects = []
        for ts in timestamps[:-1]:  # One fewer distance than messages
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                datetime_objects.append(dt)
            except (ValueError, AttributeError):
                # Use a placeholder for invalid timestamps
                datetime_objects.append(None)
        
        # Filter out None values
        valid_points = [(dt, dist) for dt, dist in zip(datetime_objects, distances) if dt is not None]
        if not valid_points:
            logger.warning("No valid timestamp-distance pairs for visualization")
            return
            
        plot_dts, plot_distances = zip(*valid_points)
        
        try:
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot distances
            plt.plot(plot_dts, plot_distances, '-o', markersize=2, alpha=0.7)
            
            # Mark thresholds
            plt.axhline(y=p90, color='yellow', linestyle='--', alpha=0.7, label=f'90th percentile: {p90:.3f}')
            plt.axhline(y=p95, color='orange', linestyle='--', alpha=0.7, label=f'95th percentile: {p95:.3f}')
            plt.axhline(y=p99, color='red', linestyle='--', alpha=0.7, label=f'99th percentile: {p99:.3f}')
            
            # Mark topic boundaries if provided
            if topic_segments:
                boundary_times = []
                for segment in topic_segments:
                    if segment.get("boundary_type") == "semantic" and segment.get("start_timestamp"):
                        try:
                            dt = datetime.fromisoformat(segment["start_timestamp"].replace('Z', '+00:00'))
                            boundary_times.append(dt)
                        except (ValueError, AttributeError):
                            continue
                
                # Plot vertical lines at boundary points
                for bt in boundary_times:
                    plt.axvline(x=bt, color='green', linestyle='-', alpha=0.5)
            
            # Format the plot
            plt.title("Semantic Distances Between Consecutive Messages")
            plt.xlabel("Time")
            plt.ylabel("Cosine Distance")
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_path)
            plt.close()  # Close the figure to free memory
            logger.info(f"Saved semantic distances visualization to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            plt.close()  # Ensure figure is closed even if there's an error
            return None

    def visualize_semantic_timeline(self, topic_segments: List[Dict[str, Any]], 
                                  output_path: str = "semantic_timeline.png"):
        """
        Generate a specialized timeline visualization for semantic topic boundaries
        
        This visualization focuses on showing the flow of topics over time with
        clear boundary markers and information about semantic distances.
        
        Args:
            topic_segments: List of topic segments with semantic boundary information
            output_path: Path to save the visualization
        """
        if not topic_segments:
            logger.warning("No topic segments to visualize")
            return
            
        # Prepare data for visualization
        segments_data = []
        
        for segment in topic_segments:
            # Parse timestamps
            try:
                start_time = datetime.fromisoformat(segment["start_timestamp"].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(segment["end_timestamp"].replace('Z', '+00:00'))
                
                segments_data.append({
                    "segment_id": segment.get("topic_id", 0),
                    "start": start_time,
                    "end": end_time,
                    "duration": (end_time - start_time).total_seconds() / 60,  # in minutes
                    "message_count": segment.get("message_count", 0),
                    "boundary_distance": segment.get("boundary_distance", 0)
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing timestamps for segment: {e}")
                continue
                
        if not segments_data:
            logger.warning("No valid segment data for visualization")
            return
            
        # Create DataFrame for easier plotting
        df = pd.DataFrame(segments_data)
        
        try:
            # Create the plot
            plt.figure(figsize=(14, 8))
            
            # Create a sequential colormap for segments based on their ID
            cmap = plt.cm.viridis
            segment_colors = cmap(np.linspace(0, 1, len(df)))
            
            # Plot each segment as a horizontal bar
            for i, row in df.iterrows():
                plt.barh(
                    0,  # All segments on same row for timeline view
                    width=(row["end"] - row["start"]).total_seconds() / 60,  # width in minutes
                    left=mdates.date2num(row["start"]), 
                    color=segment_colors[i],
                    alpha=0.7,
                    height=0.6
                )
                
                # Add segment ID text in the middle of each segment
                segment_mid_point = mdates.date2num(row["start"]) + (mdates.date2num(row["end"]) - mdates.date2num(row["start"])) / 2
                plt.text(
                    segment_mid_point, 
                    0, 
                    f"#{row['segment_id']}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white',
                    fontweight='bold'
                )
                
                # Add vertical line at segment boundary with distance information
                if i > 0:  # Skip the first segment (no boundary)
                    # Add vertical line at the start of the segment
                    plt.axvline(
                        x=row["start"], 
                        color='red', 
                        linestyle='-', 
                        alpha=0.8,
                        linewidth=2
                    )
                    
                    # Add boundary distance as text
                    if "boundary_distance" in row:
                        plt.text(
                            row["start"], 
                            0.3, 
                            f"d={row['boundary_distance']:.3f}", 
                            rotation=90, 
                            verticalalignment='bottom',
                            color='red',
                            fontsize=8
                        )
            
            # Format the x-axis to show times
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gcf().autofmt_xdate()
            
            # Remove y-axis ticks and labels
            plt.yticks([])
            
            # Add a colorbar legend for segment IDs
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(df) - 1))
            sm.set_array([])
            
            # Only add colorbar if we have more than one segment
            if len(df) > 1:
                try:
                    cbar = plt.colorbar(sm, orientation='horizontal', pad=0.2, aspect=40)
                    cbar.set_label('Segment ID')
                except Exception as e:
                    logger.warning(f"Error creating colorbar: {e}")
            
            # Add labels and title
            plt.title("Semantic Topic Timeline")
            plt.xlabel("Time")
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_path)
            plt.close()  # Close the figure to free memory
            logger.info(f"Saved semantic timeline visualization to {output_path}")
            
            # Create a second visualization showing message count by segment
            try:
                plt.figure(figsize=(10, 6))
                
                # Use segment ID as index
                segment_ids = [f"#{int(i)}" for i in df["segment_id"]]
                
                # Plot message counts
                plt.bar(segment_ids, df["message_count"], color=segment_colors)
                plt.title("Messages per Semantic Segment")
                plt.xlabel("Segment ID")
                plt.ylabel("Message Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save message count chart
                count_path = output_path.replace(".png", "_counts.png")
                plt.savefig(count_path)
                plt.close()  # Close the figure to free memory
                logger.info(f"Saved segment count visualization to {count_path}")
            except Exception as e:
                plt.close()  # Ensure figure is closed even if there's an error
                logger.warning(f"Error creating message count visualization: {e}")
            
            return output_path
        except Exception as e:
            plt.close()  # Ensure figure is closed even if there's an error
            logger.error(f"Error creating timeline visualization: {e}")
            return None

    def summarize_topics(self, topic_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create detailed summaries for each topic segment including representative messages
        
        Args:
            topic_segments: List of topic segments with messages
            
        Returns:
            Topic segments with added summaries
        """
        for segment in topic_segments:
            messages = segment.get("messages", [])
            
            if not messages:
                segment["summary"] = {"message_count": 0, "top_authors": [], "sample_messages": []}
                continue
                
            # Extract content for analysis
            message_texts = [msg.get("content", "") for msg in messages if msg.get("content")]
            
            # Get key participants
            authors = {}
            for msg in messages:
                author = msg.get("author_name")
                if author:
                    authors[author] = authors.get(author, 0) + 1
            
            top_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Extract key terms (simplistic approach)
            all_content = " ".join(message_texts)
            words = all_content.lower().split()
            # Filter out very short words and common words
            common_words = {'the', 'and', 'a', 'to', 'of', 'is', 'in', 'that', 'it', 'i', 
                           'this', 'you', 'for', 'are', 'on', 'was', 'have', 'with', 'be', 
                           'not', 'as', 'they', 'at', 'from', 'or', 'but', 'what', 'all', 
                           'would', 'there', 'will', 'an', 'my', 'one', 'so', 'we', 'can',
                           'if', 'by', 'about', 'me', 'just', 'has', 'do', 'like', 'how',
                           'their', 'your', 'who', 'could', 'no', 'more', 'when', 'very'}
            filtered_words = [w for w in words if len(w) > 2 and w not in common_words]
            
            # Count word frequencies
            word_count = {}
            for word in filtered_words:
                word_count[word] = word_count.get(word, 0) + 1
                
            # Get top words
            top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Select representative messages
            # 1. First message in the segment
            first_message = messages[0] if messages else None
            
            # 2. Message with the most engagement (if available)
            # Simplified approach: just use a longer message that's not the first one
            longer_messages = sorted([msg for msg in messages[1:] if msg.get("content")], 
                                    key=lambda m: len(m.get("content", "")), 
                                    reverse=True)
            representative_message = longer_messages[0] if longer_messages else None
            
            # 3. Last message in the segment
            last_message = messages[-1] if len(messages) > 1 else None
            
            # Build list of sample messages, avoiding duplicates
            sample_messages = []
            for msg in [first_message, representative_message, last_message]:
                if msg and msg.get("content") and msg.get("content") not in [m.get("content") for m in sample_messages]:
                    sample_messages.append({
                        "content": msg.get("content", ""),
                        "author": msg.get("author_name", ""),
                        "timestamp": msg.get("timestamp", "")
                    })
            
            # If we have very few samples, add more messages
            if len(sample_messages) < 3 and len(messages) > 3:
                # Try to add messages from the middle of the conversation
                middle_idx = len(messages) // 2
                for i in range(middle_idx-1, middle_idx+2):
                    if 0 <= i < len(messages) and len(sample_messages) < 5:
                        msg = messages[i]
                        content = msg.get("content", "")
                        if content and content not in [m.get("content") for m in sample_messages]:
                            sample_messages.append({
                                "content": content,
                                "author": msg.get("author_name", ""),
                                "timestamp": msg.get("timestamp", "")
                            })
            
            # Create enhanced summary
            segment["summary"] = {
                "message_count": len(messages),
                "top_authors": top_authors,
                "top_terms": top_words[:10],  # Top 10 keywords
                "start_time": segment["start_timestamp"],
                "end_time": segment["end_timestamp"],
                "topic_id": f"topic_{segment['cluster_id']}",
                "sample_messages": sample_messages,
                "message_count_by_author": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        
        return topic_segments
    
    def generate_topic_timeline_visualization(self, topic_segments: List[Dict[str, Any]], 
                                            output_path: str = "topic_timeline.png"):
        """
        Generate a timeline visualization of topic segments
        
        Args:
            topic_segments: List of topic segments with timestamps
            output_path: Path to save the visualization
        """
        if not topic_segments:
            logger.warning("No topic segments to visualize")
            return
            
        # Prepare data for visualization
        topics_data = []
        for segment in topic_segments:
            # Parse timestamps
            try:
                start_time = datetime.fromisoformat(segment["start_timestamp"].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(segment["end_timestamp"].replace('Z', '+00:00'))
                
                topics_data.append({
                    "topic_id": segment["cluster_id"],
                    "start": start_time,
                    "end": end_time,
                    "duration": (end_time - start_time).total_seconds() / 60,  # in minutes
                    "message_count": segment["message_count"]
                })
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing timestamps for segment: {e}")
                continue
                
        if not topics_data:
            logger.warning("No valid topic data for visualization")
            return
            
        # Create DataFrame for easier plotting
        df = pd.DataFrame(topics_data)
        
        try:
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Create a custom colormap
            colors = plt.cm.tab20(np.linspace(0, 1, len(df["topic_id"].unique())))
            topic_colors = {topic: colors[i] for i, topic in enumerate(sorted(df["topic_id"].unique()))}
            
            # Plot each topic segment as a horizontal bar
            for _, row in df.iterrows():
                plt.barh(
                    row["topic_id"], 
                    width=(row["end"] - row["start"]).total_seconds() / 60,  # width in minutes
                    left=mdates.date2num(row["start"]), 
                    color=topic_colors[row["topic_id"]],
                    alpha=0.7,
                    height=0.5
                )
            
            # Format the x-axis to show times
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gcf().autofmt_xdate()
            
            # Add labels and title
            plt.title("Topic Timeline")
            plt.xlabel("Time")
            plt.ylabel("Topic ID")
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()  # Close the figure to free memory
            logger.info(f"Saved topic timeline visualization to {output_path}")
            
            try:
                # Show the distribution of message counts per topic
                plt.figure(figsize=(10, 6))
                topic_counts = df.groupby("topic_id")["message_count"].sum().sort_values(ascending=False)
                
                # Plot topic distribution
                sns.barplot(x=topic_counts.index, y=topic_counts.values)
                plt.title("Message Count by Topic")
                plt.xlabel("Topic ID")
                plt.ylabel("Message Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save distribution chart
                distribution_path = output_path.replace(".png", "_distribution.png")
                plt.savefig(distribution_path)
                plt.close()  # Close the figure to free memory
                logger.info(f"Saved topic distribution visualization to {distribution_path}")
            except Exception as e:
                plt.close()  # Ensure figure is closed even if there's an error
                logger.warning(f"Error creating topic distribution visualization: {e}")
                
            return output_path
        except Exception as e:
            plt.close()  # Ensure figure is closed even if there's an error
            logger.error(f"Error creating topic timeline visualization: {e}")
            return None

    def process_channel(self, guild_id: str, channel_id: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       min_cluster_size: int = 5,
                       output_dir: str = "./topic_analysis") -> Dict[str, Any]:
        """
        Process a channel to extract, cluster, and visualize topics
        
        Args:
            guild_id: Discord guild ID
            channel_id: Discord channel ID
            start_date: Optional start date in ISO format
            end_date: Optional end date in ISO format
            min_cluster_size: Minimum size of clusters
            output_dir: Directory to save output files
            
        Returns:
            Results dictionary with analysis metadata
        """
        try:
            start_time = time.time()
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Fetch messages
            messages = self.fetch_channel_messages(
                guild_id=guild_id,
                channel_id=channel_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not messages:
                return {"error": "No messages found"}
                
            # Step 2: Cluster messages
            clustered_messages, umap_embedding, cluster_labels = self.cluster_messages(
                messages=messages,
                min_cluster_size=min_cluster_size
            )
            
            # Step 3: Identify topic boundaries
            topic_segments = self.identify_topic_boundaries(clustered_messages)
            
            # Step 4: Identify semantic boundaries
            semantic_segments = self.identify_semantic_boundaries(clustered_messages)
            
            # Step 5: Summarize topics
            topic_segments = self.summarize_topics(topic_segments)
            
            # Step 6: Generate visualizations
            output_base = f"{output_dir}/channel_{channel_id}"
            
            try:
                # Save topic timeline
                timeline_path = f"{output_base}_timeline.png"
                self.generate_topic_timeline_visualization(topic_segments, timeline_path)
                
                # Generate topic scatter plot with UMAP
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(
                    umap_embedding[:, 0], 
                    umap_embedding[:, 1], 
                    c=cluster_labels, 
                    cmap='Spectral',
                    s=5,
                    alpha=0.7
                )
                plt.colorbar(scatter)
                plt.title(f"Topic Clusters for Channel {channel_id}")
                plt.xlabel("UMAP Dimension 1")
                plt.ylabel("UMAP Dimension 2")
                scatter_path = f"{output_base}_clusters.png"
                plt.savefig(scatter_path)
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating cluster visualizations: {e}")
                plt.close()
            
            try:
                # Save semantic distances visualization
                self.visualize_semantic_distances(messages, semantic_segments, f"{output_base}_semantic_distances.png")
                
                # Save semantic timeline visualization
                self.visualize_semantic_timeline(semantic_segments, f"{output_base}_semantic_timeline.png")
            except Exception as e:
                logger.warning(f"Error creating semantic visualizations: {e}")
                plt.close()
            
            # Save results as JSON
            results = {
                "guild_id": guild_id,
                "channel_id": channel_id,
                "message_count": len(messages),
                "topic_count": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                "noise_count": int(np.sum(cluster_labels == -1)),
                "start_date": start_date,
                "end_date": end_date,
                "processing_time": time.time() - start_time,
                "topic_segments": [
                    {
                        "topic_id": int(segment["cluster_id"]) if isinstance(segment["cluster_id"], np.integer) else segment["cluster_id"],
                        "start_time": segment["start_timestamp"],
                        "end_time": segment["end_timestamp"],
                        "message_count": int(segment["message_count"]) if isinstance(segment["message_count"], np.integer) else segment["message_count"],
                        "summary": segment["summary"]
                    }
                    for segment in topic_segments
                ],
                "semantic_segments": semantic_segments,
                "datetime": datetime.now().isoformat()
            }
            
            # Save results as JSON
            try:
                results_path = f"{output_base}_results.json"
                
                def json_serialize_helper(obj):
                    """Helper for JSON serialization of datetime objects"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")
                
                with open(results_path, "w") as f:
                    json.dump(results, f, default=json_serialize_helper, indent=2)
                    
                logger.info(f"Saved results to {results_path}")
                
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                
            logger.info(f"Completed topic analysis for channel {channel_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing channel: {e}")
            raise
