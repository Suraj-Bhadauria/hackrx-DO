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
    "You are a highly intelligent and precise Q&A system. Your task is to answer the user's question based *only* on the provided 'CONTEXT'. "
    "Your answer must be as concise as possible. Do not include any introductory phrases like 'Based on the provided document...'. "
    "Provide only the direct answer to the question. "
    "If the answer is not found in the 'CONTEXT', you must respond with the single phrase: 'Answer not found in document'."
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

        # post processing
        raw_answer = chat_completion.choices[0].message.content.strip()

        phrases_to_remove = [
            "Based on the provided document, ",
            "Based on the provided context, ",
            "The answer to this question could not be found in the provided document. ",
            "Based on the provided document, an answer to this question could not be found."
        ]

        cleaned_answer = raw_answer
        for phrase in phrases_to_remove:
            cleaned_answer = cleaned_answer.replace(phrase, "")
        
        # --- NEW: Replace newline characters with a space ---
        final_answer = cleaned_answer.replace('\n', ' ').replace('*', '').replace('-','')

        return final_answer.strip()

    except Exception as e:
        print(f"Error during LLM call for question '{query}': {e}")
        return "An error occurred while generating the answer."