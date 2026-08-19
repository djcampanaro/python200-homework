
# Step 1: Setup

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("assignments_06/resources/groundwork_docs")
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
    print("TOP RETRIEVED SOURCE NODE:")
    print(f"Document Name: {node_with_score.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"The first 200 characters of the chunk text: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

# The assisstant sounded confident in each of the answers. It is surprising that 
# the first question about weekend hours has the our_story.txt document as its top 
# node. The hours can be found in the faq.txt document, so I thought that would be 
# the top node. Similarly, the alternative milk is mentioned in menu.txt, but 
# seasonal_specials.txt is listed as the top node for that answer.

# Step 5: Find a Failure

question = 'how many different are there?'

print(f"\nQ: {question}")
response = query_engine.query(question)
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Document Name: {node_with_score.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"First 200 characters of text: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)

# I asked 'how many different are there?' in an attempt to be vauge as though the user 
# forgot to enter the relevant information. I expected it to be difficult because there 
# is no context as to what the user is asking. It could be related to drink sizes, types 
# of drinks, locations, etc.
# The model guessed anyhow and responded with the number of seasonal specials currently 
# available. 
# The model still seems confident but only provides a one sentence answer without any 
# other detail. One has to be careful with AI-generated responses as they will find what 
# information they can and present it as the answer to the prompt. Giving more context 
# and double checking when dealing with sensitive or critical information is very 
# important.
# To improve this system, I would add a prompt ability for the user to ask questions and 
# clarify when the context might be unclear. I would set a threshold for the similarity 
# score, so that they system can ask for added context. I would also add a warning that 
# this is a generated chat and that all information should be evaluated by the user.

# Step 6: Reflection

# 1. It took three lines, including import statement, to build the index. One more line to 
# get to the query_engine. Using a framework is very valuable in the amount of time and 
# space saved. Understanding what the framework is doing is important, but utilizing the 
# framework, especially in a production program, is a much more efficient way to build the 
# semantic RAG.
# 2. This type of system would be valuable to a business with lots of information to parse 
# through for their customers. Building a chatbot to assist customers on a health insurance 
# site would help save time and resources, and provide quick responses to the user's queries.
# 3. One failure mode that RAG cannot solve when working correctly is being queried on 
# information that has not been made available either through training or provided 
# documentation. The system will search for the most relevant information and still present 
# an answer, but it will not be able to answer the query fully and correctly.
