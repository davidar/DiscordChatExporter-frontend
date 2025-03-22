#!/usr/bin/env python3
import argparse
import requests
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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
        score = f"{result.get('score', 0):.2f}"
        timestamp = format_timestamp(result.get("timestamp", ""))
        author_name = result.get("author_name", "Unknown User")
        content = result.get("content", "")
        
        # Format everything on a single line
        console.print(f"[dim]{timestamp}[/dim] [bold green]{author_name}:[/bold green] {content} [dim][cyan]({score})[/dim]")
    
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

def main():
    parser = argparse.ArgumentParser(description="Search Discord messages using semantic search")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results (1-100)")
    parser.add_argument("--no-full-messages", action="store_false", dest="fetch_full_messages", 
                        help="Don't fetch full message data")
    parser.add_argument("--server", default="http://localhost:21011", 
                        help="FastAPI server URL (default: http://localhost:21011)")
    parser.add_argument("--port", type=int, help="Server port (overrides port in --server)")
    parser.add_argument("--json", action="store_true", help="Show full JSON results")
    
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

if __name__ == "__main__":
    main() 