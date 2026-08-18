
# Step 1: Setup

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("../../python-200/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents

docs = SimpleDirectoryReader(docs_dir).load_data()
print(len(docs))
for doc in docs:
    print(doc.metadata['file_name'])

# Step 3: Build the Index and Query Engine

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

print('Index built successfully. Ready to answer questions.')

# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    node_with_score = response.source_nodes[0]
    print(f"Document Name: {node_with_score.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

#

# Step 5: Find a Failure


