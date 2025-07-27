# In app/llm_handler.py

from groq import AsyncGroq
from . import config

# This function is simple, robust, and gives the LLM the easiest task.
async def generate_answer_async(query: str, context: str, api_key: str) -> str:
    """
    Generates a precise answer to a SINGLE query using a Groq LLM,
    based *only* on the provided context. This version is the most reliable.
    """
    system_prompt = (
        "You are an expert Q&A system for legal and insurance documents. "
        "Your task is to provide a clear and concise answer to the user's question based *exclusively* "
        "on the provided context. Do not use any external knowledge. "
        "If the answer cannot be found in the context, you must state: "
        "'Based on the provided document, an answer to this question could not be found.'"
    )

    human_prompt = f"""
    CONTEXT:
    ---
    {context}
    ---

    QUESTION:
    {query}
    """
    try:
        # Create a client inside the function using the provided key
        client = AsyncGroq(api_key=api_key)

        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during LLM call for question '{query}': {e}")
        return "An error occurred while generating the answer."