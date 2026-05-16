# chatbot.py
# This is the brain of our chatbot.
# It manages conversations with recruiters and
# uses Groq AI to generate intelligent responses.

import os
from groq import Groq
from dotenv import load_dotenv
from search_engine import search_assessments

# Load our secret API key from the .env file
load_dotenv()
# Backup: set key directly (replace with your actual key)
# -------------------------------------------------------
# STEP A: System Prompt
# -------------------------------------------------------
SYSTEM_PROMPT = """You are an expert SHL assessment consultant helping recruiters choose the right assessments for their hiring needs.

Your job is to:
1. Understand the recruiter's hiring needs through conversation
2. Ask ONE clarifying question at a time when needed
3. Recommend specific SHL assessments based on the search results provided
4. Explain WHY each assessment fits their needs
5. Handle comparison requests and refinements
6. Politely refuse questions unrelated to SHL assessments or hiring

IMPORTANT RULES:
- ONLY recommend assessments from the search results provided to you
- NEVER make up or hallucinate assessment names or details
- Always explain recommendations in simple, non-technical language
- If the recruiter asks about something unrelated to hiring or assessments, politely say you can only help with SHL assessment recommendations
- Keep responses concise and professional
- When recommending, mention the assessment name, what it measures, duration, and why it fits

RESPONSE FORMAT:
- Be conversational and friendly
- Ask clarifying questions when the role or requirements are vague
- When you have enough information, provide clear recommendations
- Always end with an offer to refine or compare assessments"""


# -------------------------------------------------------
# STEP B: Initialize Groq client
# -------------------------------------------------------
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found! "
            "Make sure your .env file has GROQ_API_KEY=your_key"
        )
    return Groq(api_key=api_key)


# -------------------------------------------------------
# STEP C: Generate AI response
# -------------------------------------------------------
def generate_response(
    user_message: str,
    conversation_history: list,
    search_results: list
) -> str:
    client = get_groq_client()

    # Build context from search results
    if search_results:
        context = "\n\nRELEVANT SHL ASSESSMENTS FOUND:\n"
        for i, result in enumerate(search_results, 1):
            context += f"""
{i}. {result['name']}
   - Description: {result['description']}
   - Job Levels: {result['job_levels']}
   - Duration: {result['duration_minutes']} minutes
   - Type: {result['test_type']}
   - Remote Testing: {result['remote_testing']}
   - Adaptive: {result['adaptive']}
   - Relevance Score: {result['similarity_score']}
"""
        context += "\nOnly recommend assessments from the list above."
    else:
        context = "\nNo specific assessments found. Ask clarifying questions."

    # Build messages list
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Add conversation history
    messages.extend(conversation_history)

    # Add current message with context
    messages.append({
        "role": "user",
        "content": f"{user_message}\n{context}"
    })

    # Send to Groq and get response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )

    return response.choices[0].message.content


# -------------------------------------------------------
# STEP D: Main chat function
# -------------------------------------------------------
def chat(
    user_message: str,
    conversation_history: list,
    model,
    index,
    df
) -> dict:
    # Search for relevant assessments using FAISS
    search_results = search_assessments(
        query=user_message,
        model=model,
        index=index,
        df=df,
        top_k=5
    )

    # Generate AI response using Groq
    ai_response = generate_response(
        user_message=user_message,
        conversation_history=conversation_history,
        search_results=search_results
    )

    # Format top recommendations
    recommendations = []
    for result in search_results[:3]:
        recommendations.append({
            "name": result["name"],
            "description": result["description"],
            "job_levels": result["job_levels"],
            "duration_minutes": result["duration_minutes"],
            "test_type": result["test_type"],
            "remote_testing": result["remote_testing"],
            "adaptive": result["adaptive"]
        })

    return {
        "response": ai_response,
        "recommendations": recommendations
    }


# -------------------------------------------------------
# STEP E: Test the chatbot
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Testing chatbot...")
    print("=" * 50)

    # Initialize search system
    from search_engine import initialize_search_system
    model, index, df = initialize_search_system()

    # Simulate a conversation
    history = []

    test_messages = [
        "I need to hire a software engineer",
        "They should be good at logical thinking and problem solving",
        "It is a senior level role"
    ]

    for message in test_messages:
        print(f"\n Recruiter: {message}")
        print("-" * 40)

        result = chat(message, history, model, index, df)

        print(f" Assistant: {result['response']}")

        # Add to history for next turn
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result['response']})

    print("\n Chatbot test complete!")