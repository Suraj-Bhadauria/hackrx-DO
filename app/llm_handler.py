from groq import AsyncGroq
from . import config


# Initialize the Groq client once
client = AsyncGroq(api_key=config.GROQ_API_KEY)

async def generate_answer_async(query: str, context: str) -> str:
    """
    Generates a precise answer to a query using a Groq LLM,
    based *only* on the provided context.
    This version is asynchronous

    Args:
        query: The user's question.
        context: The relevant text chunks retrieved from the vector database.

    Returns:
        A string containing the generated answer.
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
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model=config.LLM_MODEL,
            temperature=0,  # 0 for deterministic, factual answers
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "An error occurred while generating the answer."