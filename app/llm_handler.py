# In app/llm_handler.py

from groq import AsyncGroq, GroqError, RateLimitError
from . import config
import re

async def generate_answer_async(query: str, context: str, api_key: str) -> str:
    """
    Generates clean, natural answers without formatting artifacts.
    """
    # Analyze query type for tailored response
    query_lower = query.lower()
    is_yes_no = any(query_lower.startswith(word) for word in ['is', 'are', 'does', 'do', 'can', 'will', 'would', 'should'])
    
    if is_yes_no:
        specific_instruction = "Start with 'Yes' or 'No' and explain why in simple terms."
    elif any(word in query_lower for word in ['how much', 'how many', 'cost', 'price', 'amount']):
        specific_instruction = "State the exact amount or number clearly."
    elif any(word in query_lower for word in ['what is', 'what are', 'define']):
        specific_instruction = "Provide a clear, simple definition."
    else:
        specific_instruction = "Give a direct, clear answer."

    system_prompt = f"""You are an insurance expert who explains policy details in simple, conversational language.

INSTRUCTIONS:
1. Read the CONTEXT carefully to find information that answers the QUESTION
2. Write your answer in natural, flowing sentences
3. {specific_instruction}
4. Use simple language that anyone can understand
5. Remove all technical codes, clause numbers, and legal references
6. If information is not in the context, say "Answer not found in document"

FORMATTING RULES:
- Write in complete, natural sentences
- No bullet points, dashes, or technical codes
- No line breaks or special formatting
- Maximum 2 sentences
- Sound conversational and helpful

EXAMPLE GOOD ANSWERS:
- "No, cosmetic surgery is not covered unless it's for reconstruction after an accident or burn."
- "Yes, dental treatment is covered up to $5,000 per year."
- "The waiting period is 30 days for illness-related claims."

OUTPUT ONLY YOUR FINAL ANSWER - NO LABELS OR EXPLANATIONS."""

    human_prompt = f"""CONTEXT:
{context}

QUESTION: {query}"""
    
    try:
        client = AsyncGroq(api_key=api_key)

        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=100,  # Keep it concise
        )
        
        raw_answer = chat_completion.choices[0].message.content.strip()
        
        # ENHANCED POST-PROCESSING to remove formatting artifacts
        final_answer = raw_answer.strip()
        
        # Remove newlines and replace with spaces
        final_answer = final_answer.replace('\n', ' ').replace('\r', ' ')
        
        # Remove multiple spaces
        final_answer = ' '.join(final_answer.split())
        
        # Remove technical codes and references
        import re
        # Remove patterns like "Code- Excl08:", "Clause 4.2:", etc.
        final_answer = re.sub(r'Code-?\s*[A-Za-z0-9]+:?\s*', '', final_answer)
        final_answer = re.sub(r'Clause\s+\d+(\.\d+)?:?\s*', '', final_answer)
        final_answer = re.sub(r'Section\s+\d+(\.\d+)?:?\s*', '', final_answer)
        final_answer = re.sub(r'Article\s+\d+(\.\d+)?:?\s*', '', final_answer)
        
        # Remove any remaining process labels
        cleanup_patterns = [
            "EVIDENCE:", "ANSWER:", "FINAL ANSWER:", "RESPONSE:",
            "Based on the context:", "According to the document:",
            "The evidence shows:", "From the context:", "The policy states:"
        ]
        
        for pattern in cleanup_patterns:
            if pattern in final_answer:
                parts = final_answer.split(pattern)
                final_answer = parts[-1].strip()
        
        # Clean up currency formatting issues
        # Remove the /- symbol that appears after amounts
        final_answer = re.sub(r'/-\s*', ' ', final_answer)
        
        # Standardize currency formats
        # Convert "Rs. 10,000 " to "Rs. 10,000" (remove extra spaces)
        final_answer = re.sub(r'Rs\.\s*(\d+(?:,\d+)*)\s+', r'Rs. \1 ', final_answer)
        
        # Convert "₹ 10,000 " to "Rs. 10,000"
        final_answer = re.sub(r'₹\s*(\d+(?:,\d+)*)', r'Rs. \1', final_answer)
        
        # Clean up extra spaces around numbers
        final_answer = re.sub(r'\s+', ' ', final_answer)
        
        # Ensure it's a complete sentence
        if final_answer and not final_answer.endswith(('.', '!', '?')):
            final_answer += '.'
        
        # Final cleanup - remove any remaining artifacts
        final_answer = final_answer.strip()
        
        return final_answer

    except RateLimitError:
        return "The server is busy, please try again in a moment."
    except GroqError as e:
        print(f"Groq API error for '{query}': {e}")
        return "An error occurred while communicating with the AI service."
    except Exception as e:
        print(f"Unexpected error for '{query}': {e}")
        return "An unexpected error occurred while generating the answer."