"""Tool schema. The first sentence stays within 60 characters."""

CHROMA_RECALL = {
    "name": "chroma_recall",
    "description": (
        "Recall a past decision from your Chroma collection. "
        "Search the chroma collection you already keep and return the nearest passages "
        "about what you decided to remember, including authentication. "
        "Bring back the passages I should remember."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Words from the decision you want back.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
