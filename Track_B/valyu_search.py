"""
Valyu Search Tool using the direct Valyu SDK.
Provides deep web + proprietary search for all agents.
"""

import os
from valyu import Valyu
from langchain_core.tools import tool
import dotenv


dotenv.load_dotenv()

# Initialize Valyu client
valyu_client = Valyu(api_key=os.getenv("VALYU_API_KEY"))

@tool
def valyu_search(query: str) -> str:
    """Use Valyu deep search to retrieve real-time information across
    web + proprietary sources. Returns summarized search results."""
    
    try:
        print(f"🔍 Valyu searching: {query}")
        
        # Use the direct Valyu SDK
        response = valyu_client.search(query)
        
        if not response or not response.results:
            return "No results found from Valyu search."

        # Format results
        formatted = []
        for i, result in enumerate(response.results, 1):
            formatted.append(
                f"{i}. **{getattr(result, 'title', 'Untitled')}**\n"
                f"   URL: {getattr(result, 'url', 'No URL')}\n"
                f"   Content: {getattr(result, 'content', 'No content')[:200]}...\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        error_msg = f"❌ Valyu search failed: {str(e)}"
        print(error_msg)
        return error_msg