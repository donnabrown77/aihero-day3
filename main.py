import re
import io
import zipfile
import requests
import frontmatter
import os
import numpy as np
import asyncio
from tqdm.auto import tqdm
from minsearch import Index
from sentence_transformers import SentenceTransformer
from minsearch import VectorSearch
from typing import List, Any
from pydantic_ai import Agent
from groq import Groq
from groq import AsyncGroq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from groq import DefaultAioHttpClient 




def read_repo_data(repo_owner, repo_name):
    """
    Download and parse all markdown files from a GitHub repository.

    Args:
        repo_owner: GitHub username or organization
        repo_name: Repository name

    Returns:
        List of dictionaries containing file content and metadata
    """
    prefix = "https://codeload.github.com"
    url = f"{prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main"

    print(f"Downloading repository from {url}")
    resp = requests.get(url)

    if resp.status_code != 200:
        raise Exception(f"Failed to download repository: {resp.status_code}")

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename
        filename_lower = filename.lower()

        if not (filename_lower.endswith(".md") or filename_lower.endswith(".mdx")):
            continue

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode("utf-8", errors="ignore")
                post = frontmatter.loads(content)
                data = post.to_dict()
                data["filename"] = filename
                repository_data.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    zf.close()
    repository_data[1]
    return repository_data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i+size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break

    return result


docs = read_repo_data('CatsMiaow', 'github-copilot')

chunks = []

for doc in docs:
    doc_copy = doc.copy()
    doc_content = doc_copy.pop('content')
    chunks = sliding_window(doc_content, 2000, 1000)
    for chunk in chunks:
        chunk.update(doc_copy)
    chunks.extend(chunks)

index = Index(
    text_fields=["chunk", "title", "description", "filename"],
    keyword_fields=[]
)

index.fit(chunks)



# 1. Text search
query = 'Use when errors occur in execution.'
results = index.search(query)

faq = read_repo_data('CatsMiaow', 'github-copilot')

data_faq = [d for d in faq if 'skills' in d['filename']]


faq_index = Index(
    text_fields=["description", "content"],
    keyword_fields=[]
)

faq_index.fit(data_faq)


# 2. Vector search

embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')
record = data_faq[2]


text = record['name'] + ' ' + record['content']
v_doc = embedding_model.encode(text)

query = 'Minimum Changes'
v_query = embedding_model.encode(query)
similarity = v_query.dot(v_doc)


faq_embeddings = []

for d in tqdm(data_faq):
    text = d['name'] + ' ' + d['content']
    v = embedding_model.encode(text)
    faq_embeddings.append(v)

faq_vindex = VectorSearch()
faq_vindex.fit(faq_embeddings, data_faq)
query = 'traces bugs backward through call stack'
q = embedding_model.encode(query)
results = faq_vindex.search(q)

embeddings = []
for d in tqdm(chunks):
    v = embedding_model.encode(d['chunk'])
    embeddings.append(v)

embeddings = np.array(embeddings)

vindex = VectorSearch()
vindex.fit(embeddings, chunks)

# 3. Hybrid search'

text_results = faq_index.search(query, num_results=5)

q = embedding_model.encode(query)
vector_results = faq_vindex.search(q, num_results=5)

final_results = text_results + vector_results
print(text_results)


system_prompt = """
You are a helpful assistant that searches Github repositories. 

Always search for relevant information before answering. 
If the first search doesn't give you enough information, try different search terms.

Make multiple searches if needed to provide comprehensive answers.
"""

def text_search(query: str) -> List[Any]:
    """
    Perform a text-based search on the FAQ index.

    Args:
        query (str): The search query string.

    Returns:
        List[Any]: A list of up to 5 search results returned by the FAQ index.
    """
    return faq_index.search(query, num_results=5)


def vector_search(query):
    q = embedding_model.encode(query)
    return faq_vindex.search(q, num_results=5)

def hybrid_search(query):
    text_results = text_search(query)
    vector_results = vector_search(query)
    
    # Combine and deduplicate results
    seen_ids = set()
    combined_results = []

    for result in text_results + vector_results:
        if result['filename'] not in seen_ids:
            seen_ids.add(result['filename'])
            combined_results.append(result)
    
    return combined_results


results = hybrid_search(query)
# print(results)



# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

question = query

chat_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
]




async def run_groq_agent_task(prompt: str, model_name: str = "llama-3.3-70b-versatile"):
    """
    An async function to interact with the Groq API.
    """
    # Use DefaultAioHttpClient for async operations
    async with AsyncGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        http_client=DefaultAioHttpClient(), 
    ) as client:
        # Perform an async chat completion
        completion =  await client.chat.completions.create  ( 
            model=model_name,
            messages=chat_messages,
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "text_search",
                        "description": "Perform a text-based search on the FAQ index.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"}
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]
        )
        return completion.choices[0].message.content 

async def main():
    """
    Main entry point to run our async agent tasks.
    """
    # Example of running a single task
    response = await run_groq_agent_task(question) 
    print(f"Groq Response: {response}")


# Start the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main()) 
