from fastapi import APIRouter, HTTPException
from typing import Optional
import logging
from datetime import datetime
import dateutil.parser

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ..messages.get_messages import get_messages_cursor_pagination

# Add imports for new embedding model
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="",
    tags=["guild"]
)

# Initialize the new embedding model once
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model"""
    global embedding_model
    if embedding_model is None:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model

def create_embedding_with_instruction(text, task_instruction="Represent this Discord message for semantic similarity search"):
    """Create an embedding with instruction prefix as recommended for the model"""
    # Format following the model's recommended approach
    text_with_instruction = f"Instruct: {task_instruction}\nQuery: {text}"
    return text_with_instruction

def create_window_embedding(messages, window_indices, model):
    """
    Create an embedding for a window of messages with improved context handling
    
    Args:
        messages: List of message objects
        window_indices: Indices of messages to include in window
        model: Embedding model
    
    Returns:
        Embedding vector for the combined window
    """
    if not window_indices:
        return None
    
    # Extract all message texts in the window
    window_texts = []
    for idx in window_indices:
        if idx < 0 or idx >= len(messages):
            continue
            
        message = messages[idx]
        author_name = ""
        if "author" in message:
            if isinstance(message["author"], dict):
                author_name = message["author"].get("username", "") or message["author"].get("name", "")
        
        # Extract message content
        message_content = ""
        if "content" in message:
            if isinstance(message["content"], list):
                content_items = [item.get("content", "") for item in message["content"] if "content" in item]
                message_content = " ".join(content_items)
            elif isinstance(message["content"], str):
                message_content = message["content"]
        
        if message_content:
            # Add username context
            formatted_text = f"{author_name}: {message_content}"
            window_texts.append(formatted_text)
    
    if not window_texts:
        return None
    
    # Combine all window texts into a single context unit
    # This approach gives better contextual awareness by considering the window as a whole
    combined_text = " ".join(window_texts)
    
    # Add instruction prefix as recommended for the model
    instructed_text = create_embedding_with_instruction(combined_text)
    
    # Generate embedding
    embedding = model.encode(instructed_text, convert_to_tensor=True)
    return embedding

def detect_topic_boundaries(similarities, message_ids, timestamps=None, threshold_method="auto", smooth_k=1, depth_factor=1.0):
    """
    Detect topic boundaries using improved TextTiling algorithm with timestamp awareness
    
    Args:
        similarities: List of similarity scores between adjacent windows
        message_ids: List of tuples (current_id, next_id) corresponding to each similarity score
        timestamps: List of tuples (curr_ts, next_ts) of message timestamps for detecting time gaps
        threshold_method: Method to determine boundary threshold ('auto', 'mean', 'median')
        smooth_k: Smoothing factor for the depth score curve
        depth_factor: Multiplicative factor for determining significant depth scores
        
    Returns:
        List of objects containing boundary information
    """
    if len(similarities) < 3:
        # Not enough data points for boundary detection
        return [(message_ids[i][0], message_ids[i][1], similarities[i]) for i in range(len(similarities))]
    
    # Convert similarity scores to distances
    distances = [1.0 - sim for sim in similarities]
    
    # Calculate initial similarity scores (will be used later for scaling)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Calculate time gaps between messages if timestamps are provided
    time_gap_factors = []
    time_gap_minutes = []
    if timestamps and len(timestamps) == len(message_ids):
        # Calculate all time deltas in minutes
        time_deltas = []
        for curr_ts, next_ts in timestamps:
            if curr_ts and next_ts:
                # Calculate time difference in minutes
                delta_minutes = (next_ts - curr_ts).total_seconds() / 60.0
                time_deltas.append(delta_minutes)
                time_gap_minutes.append(delta_minutes)
            else:
                time_deltas.append(0)
                time_gap_minutes.append(0)
                
        # Find significant time gaps
        # Normalize time gaps to a 0-1 scale for combination with semantic distances
        if time_deltas:
            # Find mean and std of time gaps
            mean_gap = np.mean(time_deltas)
            std_gap = np.std(time_deltas)
            
            # Calculate normalized time gaps
            for delta in time_deltas:
                if std_gap > 0:
                    # Calculate z-score of time gap
                    z_score = (delta - mean_gap) / std_gap
                    
                    # Map to 0-1 range with a steeper curve for significant gaps
                    # This gives more weight to unusually long gaps
                    if z_score > 0:
                        # Positive z-scores (gaps larger than average)
                        gap_factor = min(1.0, 0.5 + 0.5 * (1 - np.exp(-z_score / 2)))
                    else:
                        # Negative z-scores (gaps shorter than average)
                        gap_factor = max(0.0, 0.5 * np.exp(z_score / 2))
                else:
                    gap_factor = 0.5  # Default if no variation
                
                time_gap_factors.append(gap_factor)
    
    # 1. Compute smoothed curve to reduce noise
    smoothed_distances = gaussian_filter1d(distances, sigma=smooth_k)
    
    # 2. Calculate depth scores using improved DeepTiling approach
    # This is the key TextTiling metric for boundary detection
    clip = 2  # Clip value for depth calculation
    depth_scores = []
    
    # Process each gap score - skipping first and last few based on clip value
    for i in range(clip, len(smoothed_distances) - clip):
        # Find left peak (looking backward)
        lpeak = smoothed_distances[i]
        for score in smoothed_distances[i::-1]:
            if score >= lpeak:
                lpeak = score
            else:
                break
                
        # Find right peak (looking forward)
        rpeak = smoothed_distances[i]
        for score in smoothed_distances[i:]:
            if score >= rpeak:
                rpeak = score
            else:
                break
        
        # Calculate depth as average difference between peaks and current point
        valley_depth = 0.5 * (lpeak + rpeak - 2 * smoothed_distances[i])
        depth_scores.append(valley_depth)
    
    # Add zeros at the endpoints where depth score isn't defined
    full_depth_scores = [0.0] * clip + depth_scores + [0.0] * clip
    
    # 3. Determine threshold for significant boundaries
    if threshold_method == "auto":
        # Adaptive threshold based on distribution of depth scores
        non_zero_depths = [d for d in depth_scores if d > 0]
        if not non_zero_depths:
            threshold = 0.1  # Default if no valid depths
        else:
            depth_mean = np.mean(non_zero_depths)
            depth_std = np.std(non_zero_depths)
            threshold = depth_mean + (depth_std * depth_factor)
    elif threshold_method == "mean":
        threshold = np.mean(depth_scores) * depth_factor
    elif threshold_method == "median":
        threshold = np.median(depth_scores) * depth_factor
    else:
        threshold = 0.1  # Default fallback
    
    # 4. Apply threshold to identify potential boundaries
    initial_boundaries = []
    for i in range(len(full_depth_scores)):
        is_boundary = full_depth_scores[i] >= threshold
        initial_boundaries.append(is_boundary)
    
    # 5. Post-process to avoid nearby boundaries
    processed_boundaries = initial_boundaries.copy()
    
    # Ensure boundaries are not too close together
    for i in range(clip, len(initial_boundaries) - clip):
        # If multiple boundaries in this window, keep only the strongest one
        window = initial_boundaries[i-clip:i+clip+1]
        if sum(window) > 1:
            # Get depth scores for this window
            window_depths = full_depth_scores[i-clip:i+clip+1]
            
            # Find strongest boundary (highest depth)
            strongest_idx = np.argmax(window_depths)
            
            # Create new window with only the strongest boundary
            new_window = [False] * len(window)
            new_window[strongest_idx] = True
            
            # Replace window in processed boundaries
            for j in range(len(window)):
                processed_boundaries[i-clip+j] = new_window[j]
    
    # 6. Prepare final output
    message_boundaries = []
    
    for i in range(len(message_ids)):
        current_id, next_id = message_ids[i]
        
        # Raw distance (1 - similarity)
        raw_distance = distances[i]
        
        # Get depth score for this position
        depth_score = full_depth_scores[i] if i < len(full_depth_scores) else 0.0
        
        # Get time gap factor if available
        time_factor = time_gap_factors[i] if time_gap_factors and i < len(time_gap_factors) else 0.5
        
        # Get actual time gap in minutes
        time_gap = time_gap_minutes[i] if time_gap_minutes and i < len(time_gap_minutes) else 0
        
        # Is this a significant boundary based on semantic content?
        is_semantic_boundary = processed_boundaries[i] if i < len(processed_boundaries) else False
        
        # Is this a significant boundary based on time gap?
        is_time_boundary = time_factor >= 0.8  # High threshold for time gaps
        
        # Combined boundary detection
        is_boundary = is_semantic_boundary or is_time_boundary
        
        # Calculate normalized distance (0-1 scale for frontend)
        if std_dist > 0:
            z_score = (raw_distance - mean_dist) / std_dist
            normalized_z = max(0, min(1, 0.5 + (z_score * 0.15)))  # Map z-scores to 0-1 range
        else:
            normalized_z = 0.5  # Default if no variation
        
        # Calculate final score using both semantic and temporal metrics
        if is_boundary:
            if is_time_boundary:
                # Time boundaries get high scores
                time_boost = time_factor * 0.3  # Scale factor for time influence
                final_distance = max(0.5, min(1.0, 0.7 + time_boost + depth_score / (depth_score + 0.5)))
            else:
                # Semantic boundaries (as before)
                final_distance = max(0.4, min(1.0, 0.7 + depth_score / (depth_score + 0.3)))
        else:
            # Non-boundaries get a much lower score (0.0-0.3 range)
            # But still influenced by time gaps for subtle visual cues
            final_distance = min(0.3, normalized_z * 0.5 + time_factor * 0.1)
        
        # Create a justification string that explains the factors
        justification_parts = []
        
        # Add semantic info
        if raw_distance > mean_dist + std_dist:
            justification_parts.append(f"High semantic difference: {raw_distance:.2f}")
        elif raw_distance > mean_dist:
            justification_parts.append(f"Moderate semantic difference: {raw_distance:.2f}")
        
        # Add depth score info if significant
        if depth_score > 0:
            # Only call it a significant boundary if the depth score is actually meaningful
            if is_semantic_boundary and depth_score > threshold * 0.8:  # Substantial depth
                justification_parts.append(f"Strong depth score: {depth_score:.3f}")
                if depth_score > threshold:
                    justification_parts.append("Significant semantic boundary")
            elif depth_score > threshold * 0.5:  # Moderate depth
                justification_parts.append(f"Moderate depth score: {depth_score:.3f}")
            elif depth_score > 0.001:  # Minimal but non-zero depth
                justification_parts.append(f"Minor depth score: {depth_score:.3f}")
        
        # Add time gap info if available
        if time_gap > 0:
            if time_gap < 5:
                time_desc = f"Time gap: {time_gap:.1f}m"
            elif time_gap < 60:
                time_desc = f"Time gap: {time_gap:.0f}m"
            elif time_gap < 1440: # less than a day
                time_desc = f"Time gap: {time_gap/60:.1f}h"
            else:
                time_desc = f"Time gap: {time_gap/1440:.1f}d"
                
            justification_parts.append(time_desc)
            
            if is_time_boundary:
                if time_gap > 60:  # More than an hour
                    justification_parts.append("Major time boundary")
                else:
                    justification_parts.append("Significant time boundary")
        
        # If no factors are significant, add that information
        if not justification_parts:
            justification_parts.append("Minor differences")
        
        # Join all parts into a single explanation
        justification = " | ".join(justification_parts)
        
        message_boundaries.append({
            "messageId": current_id,
            "nextMessageId": next_id,
            "distance": float(final_distance),
            "is_boundary": is_boundary,
            "justification": justification
        })
    
    return message_boundaries

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
    Get semantic distances between message windows in a channel
    
    This implements an improved DeepTiling approach to detect topic boundaries more effectively,
    with additional consideration for time gaps between messages
    """
    try:
        # Calculate an expanded limit to ensure we have context for boundaries at page edges
        window_size = 4  # Increased from 3 for better context (typical DeepTiling uses 4-6)
        expanded_limit = limit + (window_size * 2)
        
        # Use the same pagination logic as messages endpoint but with expanded limit
        message_data = await get_messages_cursor_pagination(
            guild_id=guild_id,
            channel_id=channel_id,
            prev_page_cursor=prev_page_cursor,
            next_page_cursor=next_page_cursor,
            around_page_cursor=around_page_cursor,
            limit=expanded_limit  # Fetch extra messages to ensure coverage
        )
        
        # Extract all messages including our extra context messages
        all_messages = message_data.get("messages", [])
        
        if len(all_messages) < 2 * window_size:  # Need enough messages for meaningful windows
            # Not enough messages to calculate distances
            return {
                "prev_page_cursor": message_data.get("prev_page_cursor"),
                "messageDistances": [],
                "next_page_cursor": message_data.get("next_page_cursor")
            }
        
        # Track the core messages (without the extra padding)
        core_message_ids = set()
        if len(all_messages) > limit:
            # Calculate middle section that represents the actual requested page
            start_idx = 0
            if len(all_messages) > expanded_limit - window_size:
                # We have messages on both sides
                start_idx = window_size
            end_idx = min(len(all_messages), start_idx + limit)
            
            # Record the IDs of the core messages
            for i in range(start_idx, end_idx):
                if i < len(all_messages):
                    core_message_ids.add(all_messages[i]["_id"])
        else:
            # If we received fewer messages than requested, they're all core
            for msg in all_messages:
                core_message_ids.add(msg["_id"])
        
        # Get embedding model
        model = get_embedding_model()
        
        logger.info(f"Calculating improved DeepTiling semantic distances for {len(all_messages)} messages")
        
        # Calculate window embeddings and pairwise similarities for adjacent windows
        window_embeddings = []
        message_ids = []
        message_timestamps = []  # Store timestamps for time gap analysis
        
        # Process all potential boundary points
        for i in range(len(all_messages) - 1):
            # Get message IDs for reference
            current_msg_id = all_messages[i]["_id"]
            next_msg_id = all_messages[i + 1]["_id"]
            
            # Skip if we don't have enough context on both sides
            if i < window_size - 1 or i + window_size >= len(all_messages):
                continue
            
            # Only process if the current or next message is part of the core set
            if current_msg_id not in core_message_ids and next_msg_id not in core_message_ids:
                continue
                
            # Define before and after windows
            before_indices = list(range(i - window_size + 1, i + 1))  # window_size messages ending at current
            after_indices = list(range(i + 1, i + window_size + 1))   # window_size messages starting after current
            
            # Generate window embeddings
            before_embedding = create_window_embedding(all_messages, before_indices, model)
            after_embedding = create_window_embedding(all_messages, after_indices, model)
            
            if before_embedding is not None and after_embedding is not None:
                # Store for later processing
                window_embeddings.append((before_embedding, after_embedding))
                message_ids.append((current_msg_id, next_msg_id))
                
                # Extract timestamps for time gap analysis
                curr_ts = parse_timestamp(all_messages[i].get("timestamp", ""))
                next_ts = parse_timestamp(all_messages[i+1].get("timestamp", ""))
                message_timestamps.append((curr_ts, next_ts))
        
        # Calculate similarities between adjacent windows
        similarities = []
        
        for before_embedding, after_embedding in window_embeddings:
            # Get embeddings as numpy arrays
            before_embedding_np = before_embedding.cpu().numpy()
            after_embedding_np = after_embedding.cpu().numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                before_embedding_np.reshape(1, -1),
                after_embedding_np.reshape(1, -1)
            )[0][0]
            
            similarities.append(float(similarity))
        
        # Apply improved DeepTiling boundary detection with time gap awareness
        message_boundaries = detect_topic_boundaries(
            similarities=similarities,
            message_ids=message_ids,
            timestamps=message_timestamps,
            threshold_method="auto",
            smooth_k=2,  # Apply moderate smoothing
            depth_factor=1.2  # Slightly higher threshold for more selective boundaries (typical DeepTiling)
        )
        
        # Extract information for the response, including justification
        message_distances = []
        for boundary in message_boundaries:
            message_distances.append({
                "messageId": boundary["messageId"],
                "nextMessageId": boundary["nextMessageId"],
                "distance": boundary["distance"],
                "justification": boundary["justification"]
            })
        
        logger.info(f"Calculated {len(message_distances)} semantic distances with improved DeepTiling approach")
        
        # Use the original (non-expanded) pagination cursors for the response
        return {
            "prev_page_cursor": message_data.get("prev_page_cursor"),
            "messageDistances": message_distances,
            "next_page_cursor": message_data.get("next_page_cursor")
        }
        
    except Exception as e:
        logger.error(f"Error getting semantic distances: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_timestamp(timestamp_str):
    """Parse a timestamp string into a datetime object"""
    if not timestamp_str:
        return None
        
    try:
        # Try parsing as ISO format
        return dateutil.parser.parse(timestamp_str)
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return None
