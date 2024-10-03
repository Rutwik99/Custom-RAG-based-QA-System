import argparse
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatPerplexity
from modify_db import get_hf_embedding

load_dotenv()
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query(input_query):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_hf_embedding())
    results = db.similarity_search_with_score(query=input_query, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=input_query)
    
    chat = ChatPerplexity(temperature=0, model="llama-3.1-sonar-small-128k-online")
    response = chat.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"\n\nResponse: {response.content}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The input query.")
    args = parser.parse_args()
    query(args.query_text)