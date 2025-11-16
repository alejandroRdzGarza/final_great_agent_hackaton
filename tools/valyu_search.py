# tools/valyu_search.py

"""
Valyu Search Agent using the official langchain-valyu integration.
Provides deep web + proprietary search for all agents.
"""

import os
from langchain_core.tools import tool

# Ensure the API key is present
if "VALYU_API_KEY" not in os.environ:
    raise RuntimeError(
        "❌ VALYU_API_KEY not found in environment. "
        "Set it in your .env file: VALYU_API_KEY=your-key"
    )

@tool
def valyu_search(query: str) -> str:
    """Use Valyu deep search to retrieve real-time information across
    web + proprietary sources. Returns summarized search results."""
    tool = ValyuSearchTool()

    try:
        # This calls the official API with all signing/auth handled automatically
        results = tool._run(
            query=query,
            search_type="all",
            max_num_results=5,
            relevance_threshold=0.4,
            max_price=30.0,
        )

        if not results or not results.results:
            return "No results found."

        formatted = []
        for r in results.results:
            formatted.append(
                f"- **{r.get('title', 'Untitled')}**\n"
                f"  Source: {r.get('source')}\n"
                f"  Relevance: {r.get('relevance_score'):.2f}\n"
                f"  URL: {r.get('url')}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"❌ Valyu search failed: {str(e)}"
