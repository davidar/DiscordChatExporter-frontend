#!/usr/bin/env python3
import argparse
import requests
import json
import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    from ollama import AsyncClient
except ImportError:
    pass

def format_timestamp(timestamp_str):
    """Format a Discord timestamp string to a human-readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return timestamp_str

def search_messages(query, limit=10, fetch_full_messages=True, server_url="http://localhost:21011", use_context=True):
    """
    Search for messages using the semantic search API.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        fetch_full_messages: Whether to fetch full message data
        server_url: The URL of the FastAPI server
        use_context: Whether to use the context-aware search index
        
    Returns:
        Dictionary with search results or error message
    """
    url = f"{server_url}/api/semantic_search"
    params = {
        "query": query,
        "limit": limit,
        "fetch_full_messages": fetch_full_messages,
        "use_context": use_context
    }
    
    console = Console()
    
    try:
        response = requests.get(url, params=params)
        
        # Check for specific error message about filter argument
        if response.status_code == 500 and "Unknown arguments: ['filter']" in response.text:
            console.print(Panel(
                f"[bold red]Server Error:[/bold red] {response.text}",
                title="Semantic Search Error",
                border_style="red"
            ))
            return None
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        console.print(Panel(
            f"[bold red]Connection Error:[/bold red] Could not connect to the server at {server_url}\n\n",
            title="Connection Error",
            border_style="red"
        ))
        return None
        
    except requests.exceptions.RequestException as e:
        console.print(Panel(
            f"[bold red]Error making request:[/bold red] {str(e)}",
            title="Request Error",
            border_style="red"
        ))
        return None

def extract_conversation_fragments(results):
    """
    Extract conversation fragments from search results.
    
    Args:
        results: The search results from the API
        
    Returns:
        List of conversation fragments
    """
    conversation_fragments = []
    
    # Skip if no results
    if not results or "results" not in results or not results["results"]:
        return conversation_fragments
    
    # Extract fragments from each result
    for result_idx, result in enumerate(results["results"], 1):
        if "full_context" not in result or not result["full_context"]:
            continue
            
        # Parse the context into individual messages with authors
        messages = []
        context_lines = result["full_context"].strip().split("\n\n")
        
        # Get the result metadata
        result_metadata = {
            "result_index": result_idx,
            "vector_score": result.get("vector_score", 0),
            "rerank_score": result.get("rerank_score", 0),
            "has_reply_chain": result.get("has_reply_chain", False),
            "has_preceding_messages": result.get("has_preceding_messages", False),
            "has_embeds": result.get("has_embeds", False),
            "timestamp": result.get("timestamp", ""),
            "author_name": result.get("author_name", "Unknown User")
        }
        
        # Process each line in the context
        for i, line in enumerate(context_lines):
            if not line:
                continue
                
            # Try to extract author and content
            author = "Unknown"
            content = line
            
            if ": " in line:
                author, content = line.split(": ", 1)
            
            messages.append({
                "author": author,
                "content": content,
                "is_last": (i == len(context_lines) - 1)  # Flag if this is the result's matched message
            })
        
        # Add this fragment to our collection
        conversation_fragments.append({
            "messages": messages,
            "metadata": result_metadata
        })
    
    return conversation_fragments

def merge_conversation_fragments(fragments):
    """
    Merge overlapping conversation fragments into coherent conversations.
    
    Args:
        fragments: List of conversation fragments to merge
        
    Returns:
        List of merged conversations
    """
    # Skip if no fragments
    if not fragments:
        return []
    
    # Now group fragments into conversations by finding overlaps
    conversations = []
    
    # Simple clustering of fragments based on common messages
    processed_indices = set()
    
    for i, fragment in enumerate(fragments):
        if i in processed_indices:
            continue
            
        # Start a new conversation with this fragment
        current_conversation = {
            "messages": [],
            "matched_messages": []  # Messages with metadata
        }
        
        # Add the fragment's messages to the conversation
        for msg in fragment["messages"]:
            msg_key = f"{msg['author']}:{msg['content']}"
            
            # Check if this message is already in the conversation
            existing_msg = next((m for m in current_conversation["messages"] 
                               if f"{m['author']}:{m['content']}" == msg_key), None)
            
            if not existing_msg:
                msg_copy = msg.copy()
                current_conversation["messages"].append(msg_copy)
                
                # If this is the matched message from the result, add it to matched_messages
                if msg["is_last"]:
                    current_conversation["matched_messages"].append({
                        "message_index": len(current_conversation["messages"]) - 1,
                        "metadata": fragment["metadata"]
                    })
        
        # Look for overlapping fragments to merge into this conversation
        for j, other_fragment in enumerate(fragments):
            if j == i or j in processed_indices:
                continue
                
            # Check if there's significant overlap with this fragment
            # (at least one common message)
            has_overlap = False
            for msg in other_fragment["messages"]:
                msg_key = f"{msg['author']}:{msg['content']}"
                if any(f"{m['author']}:{m['content']}" == msg_key for m in current_conversation["messages"]):
                    has_overlap = True
                    break
            
            if has_overlap:
                # Add any new messages from this fragment
                for msg in other_fragment["messages"]:
                    msg_key = f"{msg['author']}:{msg['content']}"
                    
                    # Check if already in conversation
                    existing_msg = next((m for m in current_conversation["messages"] 
                                       if f"{m['author']}:{m['content']}" == msg_key), None)
                    
                    if not existing_msg:
                        msg_copy = msg.copy()
                        current_conversation["messages"].append(msg_copy)
                        
                        # If this is the matched message, add it to matched_messages
                        if msg["is_last"]:
                            current_conversation["matched_messages"].append({
                                "message_index": len(current_conversation["messages"]) - 1,
                                "metadata": other_fragment["metadata"]
                            })
                    else:
                        # If already exists but is a matched message in this fragment,
                        # add the metadata
                        if msg["is_last"]:
                            msg_index = current_conversation["messages"].index(existing_msg)
                            current_conversation["matched_messages"].append({
                                "message_index": msg_index,
                                "metadata": other_fragment["metadata"]
                            })
                
                processed_indices.add(j)
        
        # Add the processed fragment
        processed_indices.add(i)
        conversations.append(current_conversation)
    
    return conversations

def display_results(results, show_full_json=False, show_context=True):
    """Display search results in a nice format using rich."""
    console = Console()
    
    if not results or "results" not in results or not results["results"]:
        console.print(Panel("No results found", style="yellow"))
        return
    
    count = results.get("count", 0)
    console.print(f"Found [bold cyan]{count}[/bold cyan] results")
    
    # Only reconstruct conversations if context is enabled
    if show_context:
        # Extract conversation fragments
        fragments = extract_conversation_fragments(results)
        
        # Merge fragments into conversations
        conversations = merge_conversation_fragments(fragments)
        
        # Display each conversation
        for conv_idx, conversation in enumerate(conversations, 1):
            console.print(f"\n[bold]Conversation {conv_idx}:[/bold]")
            
            # Display each message
            for i, message in enumerate(conversation["messages"]):
                author = message["author"]
                content = message["content"]
                
                # Check if this message is a matched result
                is_match = any(mm["message_index"] == i for mm in conversation["matched_messages"])
                
                if is_match:
                    # Show the matched message with highlighting
                    # Get timestamp from first metadata entry for this message
                    matched_entry = next((mm for mm in conversation["matched_messages"] if mm["message_index"] == i), None)
                    if matched_entry:
                        timestamp = format_timestamp(matched_entry["metadata"]["timestamp"])
                        console.print(f"[dim]{timestamp}[/dim] [bold green]{author}:[/bold green] {content}")
                    else:
                        console.print(f"[bold green]{author}:[/bold green] {content}")
                    
                    # Show metadata for each matching result
                    for matched in conversation["matched_messages"]:
                        if matched["message_index"] == i:
                            metadata = matched["metadata"]
                            vector_score = f"{metadata['vector_score']:.2f}"
                            rerank_score = f"{metadata['rerank_score']:.2f}"
                            result_idx = metadata["result_index"]
                            
                            # Create context info string
                            context_info = []
                            if metadata["has_reply_chain"]:
                                context_info.append("[reply chain]")
                            if metadata["has_preceding_messages"]:
                                context_info.append("[has context]")
                            if metadata["has_embeds"]:
                                context_info.append("[has embeds]")
                            
                            context_str = " ".join(context_info)
                            timestamp = format_timestamp(metadata["timestamp"])
                            
                            # Display the metadata
                            console.print(f"[dim cyan]Result {result_idx}: (vector: {vector_score}, rerank: {rerank_score}) • {timestamp} {context_str}[/dim cyan]")
                else:
                    # Show regular message
                    console.print(f"[dim blue]{author}:[/dim blue] {content}")
            
            console.print("──" * 40)  # Separator between conversations
    
    # If there are no conversations or context is disabled, show individual results
    if not show_context or (show_context and not conversations):
        console.print("\n[bold]Individual Results:[/bold]")
        
        for idx, result in enumerate(results["results"], 1):
            timestamp = format_timestamp(result.get("timestamp", ""))
            author = result.get("author_name", "Unknown")
            content = result.get("content", "")
            
            console.print(f"[bold]{idx}.[/bold] [dim]{timestamp}[/dim] [bold green]{author}:[/bold green] {content}")
            
            # Show scores
            vector_score = result.get("vector_score")
            rerank_score = result.get("rerank_score")
            
            if vector_score is not None and rerank_score is not None:
                vector_score_str = f"{vector_score:.2f}"
                rerank_score_str = f"{rerank_score:.2f}"
                console.print(f"   [dim cyan](vector: {vector_score_str}, rerank: {rerank_score_str})[/dim cyan]")
            
            console.print()
    
    # Display detailed view for each result if requested
    if show_full_json:
        console.print("\n[bold]Detailed Results:[/bold]")
        console.print(json.dumps(results, indent=2))

def check_server_status(server_url):
    """Check if the FastAPI server is online and the database is connected."""
    console = Console()
    
    try:
        response = requests.get(f"{server_url}/api/")
        if response.status_code == 200:
            status = response.json()
            api_status = status.get("api_backend", "unknown")
            db_status = status.get("database", "unknown")
            
            if api_status == "online" and db_status == "online":
                return True
            else:
                console.print(Panel(
                    f"[bold yellow]Warning:[/bold yellow] Server status: API={api_status}, Database={db_status}\n\n"
                    "The server is reachable but might not be fully operational.",
                    title="Server Status",
                    border_style="yellow"
                ))
                return False
    except requests.exceptions.RequestException:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Could not connect to the server at {server_url}\n\n"
            "Please make sure the server is running and accessible.",
            title="Server Unreachable",
            border_style="red"
        ))
        return False
    
    return False

async def generate_summary_with_ollama(results, query, model="mistral-small"):
    """Generate a summary of search results using Ollama."""
    console = Console()
    
    if not results or "results" not in results or not results["results"]:
        console.print(Panel("No results to summarize", style="yellow"))
        return
    
    # Create messages with system prompt and user content
    messages = []
    
    # Prepare search results for the system prompt
    search_results_text = ""
    
    # Extract and merge conversation fragments
    fragments = extract_conversation_fragments(results)
    conversations = merge_conversation_fragments(fragments)
    
    # Sort conversations by relevance (highest rerank_score of any matched message)
    # For each conversation, find the highest rerank_score
    for conversation in conversations:
        max_rerank_score = float('-inf')
        for match_info in conversation["matched_messages"]:
            if match_info["metadata"]["rerank_score"] > max_rerank_score:
                max_rerank_score = match_info["metadata"]["rerank_score"]
        # Store the max score with the conversation
        conversation["max_rerank_score"] = max_rerank_score
    
    # Sort conversations by max rerank score, highest first
    conversations.sort(key=lambda x: x["max_rerank_score"], reverse=True)
    
    # Limit to top 3 conversations
    conversations = conversations[:3]
    
    # Format conversations for the summary
    message_counter = 1
    
    if conversations:
        for conv_idx, conversation in enumerate(conversations, 1):
            # Add a conversation header
            search_results_text += f"CONVERSATION {conv_idx}:\n"
            
            # Track which messages are search matches
            search_match_indices = set(mm["message_index"] for mm in conversation["matched_messages"])
            
            # Include all messages in the conversation with proper formatting
            for i, message in enumerate(conversation["messages"]):
                author = message["author"]
                content = message["content"]
                
                # Get timestamp from metadata if this is a matched message
                timestamp = ""
                if i in search_match_indices:
                    matched_entry = next((mm for mm in conversation["matched_messages"] if mm["message_index"] == i), None)
                    if matched_entry:
                        timestamp = format_timestamp(matched_entry["metadata"]["timestamp"])
                
                # Format the message text
                is_match = i in search_match_indices
                message_text = f"MESSAGE {message_counter}: "
                
                # Include timestamp for matched messages
                if timestamp:
                    message_text += f"[{timestamp}] "
                
                # Add the message content
                message_text += f"@{author}: {content}"
                
                # Mark search matches with an asterisk
                # if is_match:
                #     message_text += " *"
                
                search_results_text += message_text + "\n\n"
                message_counter += 1
            
            # Add a separator between conversations
            search_results_text += "---\n\n"
    
    # If we don't have any conversations, fall back to direct results
    if not search_results_text and "results" in results:
        # Sort results by rerank score
        sorted_results = sorted(results["results"], key=lambda x: x.get("rerank_score", 0), reverse=True)
        # Limit to top 5 results
        sorted_results = sorted_results[:5]
        
        for i, result in enumerate(sorted_results, 1):
            timestamp = format_timestamp(result.get("timestamp", ""))
            author = result.get("author_name", "Unknown User")
            content = result.get("content", "")
            search_results_text += f"MESSAGE {i}: [{timestamp}] @{author}: {content}\n\n"
    
    # If no results found
    if not search_results_text:
        console.print(Panel("No results to summarize", style="yellow"))
        return
    
    # System message for balanced, authentic style with concise responses
    system_message = {
        "role": "system", 
        "content": f"""You are the Keeper of Lore - an eccentric old-timer who's witnessed countless server discussions.

Context (your collected records):
{search_results_text}

When responding:
- One focused paragraph only
- Speak as someone who witnessed these exchanges
- Be slightly eccentric but never overly enthusiastic 
- Cut to what matters without flowery language
- Include relevant details only

Respond like someone who's seen it all before - knowledgeable but slightly jaded."""
    }
    
    messages.append(system_message)
    
    # Use the original search query as the user content
    messages.append({"role": "user", "content": query})

    print(json.dumps(messages, indent=2))
    
    try:
        console.print("\n[bold cyan]Consulting the Keeper of Lore...[/bold cyan]")
        client = AsyncClient()
        
        # Use streaming to show tokens as they're generated
        console.print("\n[dim italic]The Keeper of Lore speaks...[/dim italic]")
        async for chunk in await client.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            print(chunk['message']['content'], end='', flush=True)
        
        print("\n")  # Add a newline at the end
        
    except Exception as e:
        console.print(f"\n[bold red]Connection error:[/bold red] {str(e)}")
        console.print("Make sure Ollama is installed and running with the mistral-small model pulled.")
        console.print("You can install Ollama from https://ollama.com/ and run 'ollama pull mistral-small'")

def main():
    parser = argparse.ArgumentParser(description="Search Discord messages using semantic search")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of results (1-100)")
    parser.add_argument("--no-full-messages", action="store_false", dest="fetch_full_messages", 
                        help="Don't fetch full message data")
    parser.add_argument("--server", default="http://localhost:21011", 
                        help="FastAPI server URL (default: http://localhost:21011)")
    parser.add_argument("--port", type=int, help="Server port (overrides port in --server)")
    parser.add_argument("--json", action="store_true", help="Show full JSON results")
    parser.add_argument("--summarize", action="store_true", help="Generate a summary of results using Ollama")
    parser.add_argument("--model", default="mistral-small", help="Ollama model to use for summarization")
    parser.add_argument("--no-context", action="store_false", dest="use_context",
                        help="Don't use context-aware search index")
    parser.add_argument("--hide-context", action="store_false", dest="show_context",
                        help="Don't display message context in results")
    
    args = parser.parse_args()
    
    # Override port if specified
    if args.port:
        server_url = args.server.split(':')[0] + ':' + args.server.split(':')[1] + f":{args.port}"
    else:
        server_url = args.server
    
    # Check server status first
    console = Console()
    console.print("Checking server status...")
    if not check_server_status(server_url):
        return
    
    console.print(f"Searching for: [bold cyan]{args.query}[/bold cyan]")
    console.print(f"Using {'context-aware' if args.use_context else 'standard'} search index with client-side conversation merging")
    
    results = search_messages(
        query=args.query,
        limit=args.limit,
        fetch_full_messages=args.fetch_full_messages,
        server_url=server_url,
        use_context=args.use_context
    )
    
    if results:
        display_results(results, args.json, args.show_context)
        
        # Generate summary if requested
        if args.summarize:
            try:
                asyncio.run(generate_summary_with_ollama(results, args.query, args.model))
            except NameError:
                console.print(Panel(
                    "[bold red]Error:[/bold red] The ollama package is not installed.\n\n"
                    "Please install it with: pip install ollama\n"
                    "Then make sure Ollama is running and you've pulled the model with: ollama pull mistral-small",
                    title="Missing Dependency",
                    border_style="red"
                ))

if __name__ == "__main__":
    main()
