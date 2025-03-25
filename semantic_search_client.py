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

def search_messages(query, limit=10, fetch_full_messages=True, server_url="http://localhost:21011"):
    """
    Search for messages using the semantic search API.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        fetch_full_messages: Whether to fetch full message data
        server_url: The URL of the FastAPI server
        
    Returns:
        Dictionary with search results or error message
    """
    url = f"{server_url}/api/semantic_search"
    params = {
        "query": query,
        "limit": limit,
        "fetch_full_messages": fetch_full_messages
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

def display_results(results, show_full_json=False):
    """Display search results in a nice format using rich."""
    console = Console()
    
    if not results or "results" not in results or not results["results"]:
        console.print(Panel("No results found", style="yellow"))
        return
    
    count = results.get("count", 0)
    console.print(f"Found [bold cyan]{count}[/bold cyan] results")
    
    # Display results in a dense IRC-like format
    for result in results["results"]:
        vector_score = f"{result.get('vector_score', 0):.2f}"
        rerank_score = f"{result.get('rerank_score', 0):.2f}"
        timestamp = format_timestamp(result.get("timestamp", ""))
        author_name = result.get("author_name", "Unknown User")
        content = result.get("content", "")
        
        # Format everything on a single line with both scores
        console.print(f"[dim]{timestamp}[/dim] [bold green]{author_name}:[/bold green] {content} [dim][cyan](vector: {vector_score}, rerank: {rerank_score})[/dim]")
    
    # Display detailed view for each result if requested
    if show_full_json:
        console.print("\n[bold]Detailed Results:[/bold]")
        console.print(json.dumps(results, indent=2))
    else:
        console.print("\nUse --json to see full result details")

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
    for i, result in enumerate(results["results"], 1):
        timestamp = format_timestamp(result.get("timestamp", ""))
        author = result.get("author_name", "Unknown User")
        content = result.get("content", "")
        search_results_text += f"{i}. [{timestamp}] @{author}: {content}\n"
    
    # System message for balanced, authentic style with concise responses
    system_message = {
        "role": "system", 
        "content": f"""You are the Archivist - someone who's been around the Discord server for ages and has seen all these conversations unfold. You have a natural ability to recall discussions and their context, speaking like someone who was there, not like someone reading records.

Context (you recall these conversations naturally):
{search_results_text}

When responding:
- Deliver your response as a single, focused paragraph
- Speak from firsthand recollection, as someone who participated in or observed these conversations
- Find a balanced tone - neither too verbose nor too terse
- Include enough meaningful details to provide context, but be concise
- Use a natural, slightly world-weary tone that comes from having seen many similar discussions
- No need to cite sources or reference messages - this is knowledge you personally possess
- Occasional mild humour or insight is welcome when appropriate
- Phrase things conversationally but with substance - like a knowledgeable old-timer at a pub
- Focus on the substance of what people were discussing rather than the specific messages
- Provide thoughtful context that connects related ideas when helpful

Example (if asked about Docker issues):
"The Docker situation on Windows has been problematic lately. There was a stretch where several people hit WSL configuration issues that prevented Docker from running properly. Someone eventually discovered that updating to WSL2 before reinstalling Docker fixed most of the problems. This has been a recurring theme with Windows containerization - the WSL layer adds complexity but usually holds the key to making things work."

Respond with a focused, insightful paragraph that feels like it comes from memory, not research."""
    }
    
    messages.append(system_message)
    
    # Use the original search query as the user content
    messages.append({"role": "user", "content": query})
    
    try:
        console.print("\n[bold cyan]Asking the Archivist...[/bold cyan]")
        client = AsyncClient()
        
        # Use streaming to show tokens as they're generated
        console.print("\n[dim italic]The Archivist recalls...[/dim italic]")
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
    
    results = search_messages(
        query=args.query,
        limit=args.limit,
        fetch_full_messages=args.fetch_full_messages,
        server_url=server_url
    )
    
    if results:
        display_results(results, args.json)
        
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