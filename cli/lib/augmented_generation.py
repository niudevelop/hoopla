import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def build_llm_rag_prompt(query: str, docs: list[dict], method) -> str:
    formatted_results = []
    for i, d in enumerate(docs, start=1):
        title = d.get("title", "")
        desc = d.get("document", "")
        formatted_results.append(f"{i}. {title} - {desc}")
    match method:

        case "rag":
            return f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {chr(10).join(formatted_results)}

        Provide a comprehensive answer that addresses the query:"""

        case "summarize":
            return f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{chr(10).join(formatted_results)}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

        case "citations":
            return f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{chr(10).join(formatted_results)}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

        case "question":
            return f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{chr(10).join(formatted_results)}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
        case _:
            return None


def rag_search(query: str, docs: list):
    return model_call(query, docs, "rag")


def summarize(query: str, docs: list):
    return model_call(query, docs, "summarize")


def citations(query: str, docs: list):
    return model_call(query, docs, "citations")


def question(query: str, docs: list):
    return model_call(query, docs, "question")


def model_call(query: str, docs: list, method: str):
    prompt = build_llm_rag_prompt(query, docs, method)
    if prompt == None:
        raise Exception("Unkown RAG Command")
    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt,
    )

    return resp.text
