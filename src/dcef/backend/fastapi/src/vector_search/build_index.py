#!/usr/bin/env python3
"""
Script to build a vector index from MongoDB data
Usage: python build_index.py [--guild_id GUILD_ID] [--max_messages MAX_MESSAGES] [--no-resume] [--force-recreate]
"""

import argparse
import logging
import sys
import time
import pathlib
from typing import List, Optional
import tqdm

sys.path.append("../..") # Add fastapi directory to path

from src.common.Database import Database
from src.vector_search.vector_store import VectorStore, PERSIST_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_all_guild_ids() -> List[str]:
    """Get all guild IDs from the database"""
    try:
        collection_guilds = Database.get_global_collection("guilds")
        guilds = collection_guilds.find({}, {"_id": 1})
        return [guild["_id"] for guild in guilds]
    except Exception as e:
        logger.error(f"Error getting guild IDs: {e}")
        return []

def build_index(guild_id: Optional[str] = None, max_messages: Optional[int] = None, force_recreate: bool = False):
    """
    Build the vector index for all messages or a specific guild
    
    Args:
        guild_id: Specific guild ID to index (None = all guilds)
        max_messages: Maximum messages to index per guild (None = all messages)
        force_recreate: Whether to recreate the vector collection (for dimension changes)
    """
    try:
        # Ensure the persist directory exists
        pathlib.Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Using vector index directory: {PERSIST_DIR}")
        
        if force_recreate:
            logger.info("Force recreating vector collection to handle dimension changes...")
        
        # Initialize vector store with force_recreate if needed
        vector_store = VectorStore(force_recreate=force_recreate)
        
        if guild_id:
            # Index specific guild
            logger.info(f"Building index for guild {guild_id}")
            # If we're recreating the collection, don't resume
            result = vector_store.index_guild_messages(
                guild_id=guild_id,
                max_messages=max_messages,
            )
            logger.info(f"Indexed {result['indexed_messages']} messages in {result['elapsed_time']:.2f} seconds")
        else:
            # Index all guilds
            guild_ids = get_all_guild_ids()
            logger.info(f"Found {len(guild_ids)} guilds to index")
            
            # Progress tracker for guilds
            with tqdm.tqdm(total=len(guild_ids), desc="Indexing all guilds", unit="guild") as pbar:
                for i, gid in enumerate(guild_ids):
                    logger.info(f"Building index for guild {gid} ({i+1}/{len(guild_ids)})")
                    try:
                        # If we're recreating the collection, don't resume
                        result = vector_store.index_guild_messages(
                            guild_id=gid,
                            max_messages=max_messages,
                        )
                        logger.info(f"Indexed {result['indexed_messages']} messages in {result['elapsed_time']:.2f} seconds")
                        pbar.update(1)
                    except Exception as guild_e:
                        logger.error(f"Error indexing guild {gid}: {guild_e}")
                        # Continue with next guild instead of failing completely
                        pbar.update(1)
                        continue
    
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Build vector index for semantic search")
    parser.add_argument("--guild_id", type=str, help="Specific guild ID to index (default: all guilds)")
    parser.add_argument("--max_messages", type=int, help="Maximum messages to index per guild (default: all)")
    parser.add_argument("--force-recreate", action="store_true",
                        help="Force recreate the vector collection (for dimension changes)")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Check if MongoDB is available
    if not Database.is_online():
        logger.error("MongoDB is not online. Please start MongoDB first.")
        sys.exit(1)
    
    try:
        build_index(
            guild_id=args.guild_id, 
            max_messages=args.max_messages, 
            force_recreate=args.force_recreate
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total indexing time: {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 