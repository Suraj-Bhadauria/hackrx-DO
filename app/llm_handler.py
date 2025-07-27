from groq import AsyncGroq
from . import config
import json


# Initialize the Groq client once
client = AsyncGroq(api_key=config.GROQ_API_KEY)

# This is our new, optimized function
async def generate_all_answers_at_once_async(questions: list[str], context: str) -> list[str]:
    """
    Generates answers for a list of questions based on a single context block,
    using just ONE call to the LLM.

    Args:
        questions: A list of user questions.
        context: A single string containing all the relevant text for all questions.

    Returns:
        A list of strings, where each string is the answer to the corresponding question.
    """
    # Create a numbered list of questions to pass to the model
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    # This is a more complex prompt that tells the LLM to behave like a JSON API
    system_prompt = (
        "You are an expert Q&A system. Your task is to answer a list of questions based *exclusively* "
        "on the provided context. You must follow these rules strictly:\n"
        "1. Answer each question individually.\n"
        "2. Your response must be a single, valid JSON object.\n"
        "3. The JSON object should have a single key named 'answers'.\n"
        "4. The value of 'answers' must be a JSON array of strings.\n"
        "5. The array must contain exactly the same number of answers as there are questions.\n"
        "6. If the answer to a question cannot be found in the context, the corresponding string in the array must be: "
        "'Based on the provided document, an answer to this question could not be found.'\n"
        "7. Do not include any explanation or preamble. Your entire output must be only the JSON object."
    )

    human_prompt = f"""
    CONTEXT:
    ---
    {context}
    ---

    QUESTIONS:
    {formatted_questions}
    """

    print("Making a single, batched API call to the LLM for all questions...")
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=2048,  # Increase max_tokens for a larger combined response
            # CRITICAL: Tell the model to output JSON
            response_format={"type": "json_object"},
        )

        response_text = chat_completion.choices[0].message.content
        print("Received batched response from LLM.")

        # Parse the JSON string from the LLM's response
        response_data = json.loads(response_text)
        
        # Validate the response
        if 'answers' in response_data and isinstance(response_data['answers'], list):
            return response_data['answers']
        else:
            print("Error: LLM response was not in the expected JSON format.")
            # Return an error for every question if the format is wrong
            return ["Error parsing LLM response." for _ in questions]

    except Exception as e:
        print(f"Error during batched LLM call: {e}")
        return [f"An error occurred while generating the answer: {e}" for _ in questions]



# # async def generate_answer_async(query: str, context: str) -> str:
#     """
#     Generates a precise answer to a query using a Groq LLM,
#     based *only* on the provided context.
#     This version is asynchronous

#     Args:
#         query: The user's question.
#         context: The relevant text chunks retrieved from the vector database.

#     Returns:
#         A string containing the generated answer.
#     """
#     system_prompt = (
#         "You are an expert Q&A system for legal and insurance documents. "
#         "Your task is to provide a clear and concise answer to the user's question based *exclusively* "
#         "on the provided context. Do not use any external knowledge. "
#         "If the answer cannot be found in the context, you must state: "
#         "'Based on the provided document, an answer to this question could not be found.'"
#     )

#     human_prompt = f"""
#     CONTEXT:
#     ---
#     {context}
#     ---

#     QUESTION:
#     {query}
#     """

#     try:
#         chat_completion = await client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": human_prompt},
#             ],
#             model=config.LLM_MODEL,
#             temperature=0,  # 0 for deterministic, factual answers
#             max_tokens=1024,
#         )
#         return chat_completion.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"Error during LLM call: {e}")
#         return "An error occurred while generating the answer."