# In app/llm_handler.py

from groq import AsyncGroq, GroqError, RateLimitError
from . import config
from .rate_limiter import get_rate_limiter  # Import the rate limiter
import re
import asyncio

async def generate_answer_async(query: str, context: str, api_key: str = None, api_key_index: int = 0) -> str:
    """Generates clean, natural answers with multi-key rate limiting"""
    
    # FIXED: Use the global rate limiter properly
    if api_key is None:
        limiter = get_rate_limiter()
        api_key, key_index = await limiter.get_next_available_key()
        
    if not api_key:
        return "API key not configured properly."
    
    # Analyze query type for tailored response
    query_lower = query.lower()
    is_yes_no = any(query_lower.startswith(word) for word in ['is', 'are', 'does', 'do', 'can', 'will', 'would', 'should'])
    
    # Check if question is likely general knowledge/programming
    general_knowledge_indicators = [
        'code', 'program', 'javascript', 'js', 'python', 'html', 'css', 'sql',
        'algorithm', 'function', 'variable', 'array', 'loop', 'syntax', 'debug',
        'random number', 'calculate', 'formula', 'math', 'mathematics',
        'definition of', 'what is programming', 'how to code', 'write a function'
    ]
    
    is_general_knowledge = any(indicator in query_lower for indicator in general_knowledge_indicators)
    
    if is_yes_no:
        specific_instruction = "Start with 'Yes' or 'No' and explain why in simple terms."
    elif any(word in query_lower for word in ['how much', 'how many', 'cost', 'price', 'amount']):
        specific_instruction = "State the exact amount or number clearly."
    elif any(word in query_lower for word in ['what is', 'what are', 'define']):
        specific_instruction = "Provide a clear, simple definition."
    else:
        specific_instruction = "Give a direct, clear answer."

    # Enhanced system prompt with general knowledge capability
    if is_general_knowledge:
        system_prompt = f"""You are a helpful AI assistant with expertise in insurance and general knowledge.

INSTRUCTIONS:
1. If the QUESTION is about insurance/policy, use the CONTEXT to answer
2. If the QUESTION is about general knowledge, programming, math, or other topics, answer using your knowledge
3. {specific_instruction}
4. Use simple, clear language
5. Be direct and helpful

FORMATTING RULES:
- Write in complete, natural sentences
- No backslashes, extra quotes, or technical artifacts
- Maximum 2-3 sentences for insurance questions
- For code questions, provide clean, working code
- Sound helpful and professional

OUTPUT ONLY YOUR FINAL ANSWER - NO LABELS OR EXPLANATIONS."""
    else:
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
- No backslashes, extra quotes, or formatting artifacts
- No line breaks or special formatting
- Maximum 2 sentences
- Sound conversational and helpful

EXAMPLE GOOD ANSWERS:
- "No, cosmetic surgery is not covered unless it's for reconstruction after an accident or burn."
- "Yes, dental treatment is covered up to Rs. 5,000 per year."
- "The waiting period is 30 days for illness-related claims."

OUTPUT ONLY YOUR FINAL ANSWER - NO LABELS OR EXPLANATIONS."""

    # Adjust human prompt based on question type
    if is_general_knowledge:
        human_prompt = f"""QUESTION: {query}

Additional context (if relevant): {context[:500] if context else "No specific context provided"}"""
    else:
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
            temperature=0.1 if is_general_knowledge else 0,  # Slightly higher temp for general knowledge
            max_tokens=150 if is_general_knowledge else 100,  # More tokens for code/explanations
        )
        
        raw_answer = chat_completion.choices[0].message.content.strip()
        
        # ENHANCED POST-PROCESSING - Remove all formatting artifacts
        final_answer = raw_answer.strip()
        
        # Remove newlines and replace with spaces
        final_answer = final_answer.replace('\n', ' ').replace('\r', ' ')
        
        # CRITICAL: Remove backslash artifacts that cause noise
        final_answer = re.sub(r'\\\\', '', final_answer)  # Remove double backslashes
        final_answer = re.sub(r'\\', '', final_answer)    # Remove single backslashes
        
        # Fix quote and comma formatting issues
        final_answer = re.sub(r'\s*,\s*"', ', "', final_answer)  # Fix comma spacing before quotes
        final_answer = re.sub(r'"\s*and\s*,\s*"', '" and "', final_answer)  # Fix quote spacing
        final_answer = re.sub(r'"\s*,\s*"', '" and "', final_answer)  # Fix multiple quotes
        
        # Remove multiple spaces
        final_answer = ' '.join(final_answer.split())
        
        # Remove technical codes and references (only for insurance questions)
        if not is_general_knowledge:
            final_answer = re.sub(r'Code-?\s*[A-Za-z0-9]+:?\s*', '', final_answer)
            final_answer = re.sub(r'Clause\s+\d+(\.\d+)?:?\s*', '', final_answer)
            final_answer = re.sub(r'Section\s+\d+(\.\d+)?:?\s*', '', final_answer)
            final_answer = re.sub(r'Article\s+\d+(\.\d+)?:?\s*', '', final_answer)
            
            # Remove process labels for insurance questions
            cleanup_patterns = [
                "EVIDENCE:", "ANSWER:", "FINAL ANSWER:", "RESPONSE:",
                "Based on the context:", "According to the document:",
                "The evidence shows:", "From the context:", "The policy states:"
            ]
            
            for pattern in cleanup_patterns:
                if pattern in final_answer:
                    parts = final_answer.split(pattern)
                    final_answer = parts[-1].strip()
            
            # Clean up currency formatting
            final_answer = re.sub(r'/-\s*', ' ', final_answer)
            final_answer = re.sub(r'Rs\.\s*(\d+(?:,\d+)*)\s+', r'Rs. \1 ', final_answer)
            final_answer = re.sub(r'₹\s*(\d+(?:,\d+)*)', r'Rs. \1', final_answer)
        
        # Final cleanup - remove any remaining artifacts
        final_answer = re.sub(r'\s+', ' ', final_answer)
        
        # Ensure proper sentence ending
        if final_answer and not final_answer.endswith(('.', '!', '?', ';')):
            final_answer += '.'
        
        final_answer = final_answer.strip()
        
        print(f"✅ Generated {'general knowledge' if is_general_knowledge else 'insurance'} answer for: {query[:50]}...")
        return final_answer

    except RateLimitError as e:
        print(f"⚠️ Rate limit hit: {str(e)}")
        await asyncio.sleep(5)
        return "The server is busy, please try again in a moment."
        
    except GroqError as e:
        print(f"❌ Groq API error: {e}")
        return "An error occurred while communicating with the AI service."
        
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return "An unexpected error occurred while generating the answer."

async def answer_direct_llm(questions: list, document_context: str = None, metadata: dict = None) -> list:
    """Direct LLM answering for large/general documents with multi-key support"""
    print(f"🤖 Direct LLM processing {len(questions)} questions...")
    
    # FIXED: Import the rate limiter properly
    limiter = get_rate_limiter()
    
    answers = []
    for question in questions:
        # Build document-aware system prompt with emphasis on conciseness
        if document_context and 'constitution' in document_context.lower():
            if 'india' in document_context.lower() or 'bharat' in document_context.lower():
                system_prompt = """You are an expert on Indian constitutional law. Provide clear, concise answers about Indian legal matters.

INSTRUCTIONS:
1. Focus on Indian Constitution and Indian legal system
2. Give direct, practical answers in 1-2 sentences maximum
3. Mention specific articles only when essential
4. Use simple, conversational language
5. Be authoritative but approachable

FORMATTING RULES:
- Maximum 50 words per answer
- No repetition or redundancy  
- No legal jargon unless necessary
- Sound like a helpful legal advisor
- End with a period

EXAMPLES:
Q: "Is arrest without warrant legal?"
A: "Yes, but only in specific circumstances like catching someone committing a crime. You must be informed of the reason and have the right to legal representation under Article 22."

OUTPUT ONLY YOUR FINAL ANSWER - NO INTRODUCTIONS OR EXPLANATIONS."""

            else:
                system_prompt = """You are a constitutional law expert. Provide clear, concise answers about constitutional matters.

INSTRUCTIONS:
1. Answer based on constitutional principles
2. Give direct, practical answers in 1-2 sentences maximum  
3. Use simple, conversational language
4. Be authoritative but approachable

FORMATTING RULES:
- Maximum 50 words per answer
- No repetition or legal jargon
- Sound like a helpful legal advisor"""
        
        elif document_context and any(term in document_context.lower() for term in ['principia', 'newton', 'mathematics']):
            system_prompt = """You are a science expert. Provide clear, concise explanations of scientific concepts.

INSTRUCTIONS:
1. Explain scientific principles simply
2. Give direct answers in 1-2 sentences maximum
3. Use everyday language, avoid technical jargon
4. Be educational but approachable

FORMATTING RULES:
- Maximum 50 words per answer
- No complex equations or formulas
- Sound like a helpful science teacher"""
        
        else:
            system_prompt = """You are a knowledgeable assistant. Provide clear, concise answers based on the document.

INSTRUCTIONS:
1. Use document context to provide accurate answers
2. Give direct answers in 1-2 sentences maximum
3. Use simple, conversational language
4. If context doesn't contain the answer, say so briefly

FORMATTING RULES:
- Maximum 50 words per answer
- No repetition or unnecessary details
- Sound helpful and friendly"""

        # Build concise user prompt
        user_prompt = f"Question: {question}"
        
        if document_context:
            # Extract only the most relevant context (much shorter)
            context_limit = 800  # Reduced from 1500
            truncated_context = document_context[:context_limit]
            user_prompt += f"\n\nRelevant Context: {truncated_context}"
        
        user_prompt += "\n\nProvide a direct, concise answer in maximum 50 words:"
        
        try:
            # FIXED: Use multi-key system consistently
            api_key, key_index = await limiter.get_next_available_key()
            
            client = AsyncGroq(api_key=api_key)
            
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=config.LLM_MODEL,
                temperature=0.2,    # Slightly more creative for natural language
                max_tokens=80       # Drastically reduced from 200
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Enhanced cleanup for polished answers
            answer = re.sub(r'\s+', ' ', answer)  # Remove extra spaces
            answer = re.sub(r'\\\\', '', answer)  # Remove backslashes
            answer = re.sub(r'\\', '', answer)
            
            # Remove common verbose patterns
            verbose_patterns = [
                r'^According to.*?,\s*',
                r'^Based on.*?,\s*',
                r'^Under.*?,\s*',
                r'^As per.*?,\s*',
                r'^In accordance with.*?,\s*',
                r'It states:.*?"',
                r'This provision states:.*?"',
                r'The Constitution states:.*?"',
            ]
            
            for pattern in verbose_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            
            # Remove repetitive phrases
            repetitive_patterns = [
                r'\bor any of them\b',
                r'\b, as the case may be\b',
                r'\brespectively\b',
                r'\bunder this article\b',
                r'\bof the Constitution\b(?!\s+of\s+India)',  # Keep "Constitution of India"
            ]
            
            for pattern in repetitive_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
            
            # Clean up common legal redundancies
            answer = re.sub(r'\bArticle\s+(\d+)(?:\(\d+\))?\s+of\s+the\s+Constitution(?:\s+of\s+India)?\b', 
                          r'Article \1', answer)
            
            # Fix sentence flow
            answer = re.sub(r'\s+', ' ', answer)  # Clean extra spaces again
            answer = answer.strip()
            
            # Ensure proper capitalization
            if answer and answer[0].islower():
                answer = answer[0].upper() + answer[1:]
            
            # Ensure proper ending
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            # Final length check - truncate if still too long
            if len(answer.split()) > 60:  # If more than 60 words
                words = answer.split()
                answer = ' '.join(words[:50]) + '.'
                
            answers.append(answer)
            print(f"✅ Polished answer ({len(answer.split())} words): {question[:50]}...")
            
        except Exception as e:
            print(f"❌ Error in direct LLM: {e}")
            answers.append("I don't have enough information to answer this question.")
    
    return answers