from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---

# Concepts Q1

# Scenario A: The best approach for this scenario is RAG. The legal team wants to 
# reference a large amount of their own material, which goes beyond the training data 
# used to build models. This material will need to be broken down into chunks, embedded, 
# and indexed to provide the best results for the company.

# Scenario B: The best approach is fine-tuning. The company wants the model to speak with 
# a specific voice that can be trained using the 3,000 examples their in-house writers 
# have produced. Fine-tuning a model with give all the advantages of a LLM but in the 
# unique voice the company wishes to create.

# Scenario C: The best approach is prompt engineering. The data analyst needs a one-off 
# job done by the model and can reference the report in their prompt. There is no need 
# to build upon the model or tune it in a way that benefits this use as it will not be 
# used as such again.

# Concepts Q2

# A confidently incorrect answer is more harmful than one that acknowledges it is unsure 
# because the user is not given any indication that the information could be incorrect. 
# If they do not check and compare the info, they could move forward in a real life 
# situation that proves embarrasing or dangerous. One example where this could be harmful 
# is with medical advice. A user could self-diagnose incorrectly and/or take a medicine 
# that could cause adverse effects.

# Models are built in a way that sound very human, and their tone is friendly in a way that 
# may elicit trust with the information given. If the model sounds confident, someone who 
# is unaware of the hallucinations may trust that it is pulling the best information from 
# all its training.

# Concepts Q3

# steps = [
#     "Extract text from source documents",         - The text from all the documents from the source are collected and stored
#     "Split text into chunks",                     - The text is then split into chunks for better ingestion
#     "Convert text chunks into embeddings",        - The chunks are converted into embeddings so they can be easily compared with queries
#     "Receive the user's query",                   - A user queries the system with text
#     "Embed the user's query",                     - The user's text is embedded for comparison to the previously embedded chunks
#     "Retrieve the most relevant chunks",          - The most relevant chunks based on the cosine similarity to the embedded query text are pulled
#     "Inject retrieved chunks into the prompt",    - The relevant chunks are added to the prompt along with the query and sent to the model
#     "Generate a response from the LLM",           - The model responds using the provided information, and the results are displayed to the user
# ]

# --- Keyword RAG ---

# Keyword Q1

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

keyword = simple_keyword_retrieval(query=query, documents=documents)

# The document that was selected as the best match was 'loyalty.txt'. Once the words 
# of the query were filtered, three of the four documents had an overlap, and they 
# all tied at 1. When the scores are sorted in descending order, the loyaly document 
# ends up on top and is called as best[0].

# Keyword Q2

query = "Do you have anything without caffeine?"

keyword_2 = simple_keyword_retrieval(query=query, documents=documents)

# No document was selected, keyword RAG did not get this right because after removing all 
# stopwords, the remaining tokens did not overlap with any of the documents. Because none 
# of the overlap was greater than 0, no document can be regarded as best. Semantic RAG 
# would be a better system to use in this case. That way, the context of asking for a 
# non-caffeinated drink would have connected with the menu document.

# Keyword Q3

# I predict that like the last example, no overlapping keywords will be found. I am not seeing 
# any of the words from the question in the documents. The one that is most relevant document 
# doesn't contain the word 'rewards' which would gives the most context to the question.

query = "How do I sign up for rewards?"

keyword_2 = simple_keyword_retrieval(query=query, documents=documents)

# Yes, my prediction was correct.

# --- Semantic RAG Concepts ---

# Semantic Q1

# 1. Vector Embedding is a list of numbers assigned to a part (chunk) of a document. 
# 2. The first chunk, with a score of 0.85 is more relevant. Because it is closer to 1, it tells me 
# that it has a stronger relationship with the text to which it is being compared.
# 3. Semantic search uses cosine similarity to compare vector embeddings on a scale of -1 to 1. As, 
# the comparison gets closer to 1, the compared text and chunk are found to have a stronger relationship, 
# whether or not they contain the same words.

# Semantic Q2

# | Feature                    | Keyword RAG                       | Semantic RAG              |
# |----------------------------|-----------------------------------|---------------------------|
# | What is compared?          | Exact word overlap                | embeddings                |
# | What is retrieved?         | Full document                     | text chunks               |
# | Can it handle synonyms?    | No                                | yes                       |
# | Storage format             | Plain text dictionary             | vector store index        |
# | Relevance score            | Number of overlapping keywords    | cosine similarity -1 to 1 |

# --- Llamaindex ---

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader('../../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs').load_data()
index = VectorStoreIndex.from_documents(docs)

# Llamaindex Q1

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Yes, for the most part the retrieved chunks are relevant to the queries. The first 
# query did retrieve the Network and Data Security file, which seems less relevant 
# than the others.
# The model's response to both queries is very confident. There is no hesitation or 
# qualification in conveying the answer.
# The Network and Data Security being retrieved for the first query was unexpected. 
# Also, that the top three nodes for each query were the same, albeit in different 
# orders, was also surprising.

# Llamaindex Q2

top_ks = [1, 5]
for k in top_ks:
    query_engine_q2 = index.as_query_engine(similarity_top_k=k)
    print(f"\nQ: {questions[1]}")
    response = query_engine_q2.query(questions[1])
    print("A:", response)    

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)

# The response for k=5 is noticeably more detailed. The first example with k=1 
# provides a good answer with specifics from one document, likely one on data 
# and security. However, the second goes into even more detail, mentioning 
# credential rotation and employee training amongst other things. Generally 
# speaking, more context is better as relavent information can be found in 
# multiple documents. However, there can be a risk of reaching too deep and 
# getting unnecessary or unneeded information.

# Llamaindex Q3

query_engine_q3 = index.as_query_engine(similarity_top_k=3)

question = 'What percentage of profits go to research and development?'

print(f"\nQ: {question}")
response = query_engine_q3.query(question)
print("A:", response)   

for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

# I expected either a generic or non-answer from the response. It gave a generic 
# answer acknowledging investment in R&D, but not having any specifics in regards 
# to amounts or percentages. To change this, I could implement a threshold for 
# the similarity score. All of these scores were below .8, which feels as though 
# they are moving towards uncertainty. Maybe adding a threshold and responding by 
# asking more specifics or acknowledging a lack of information would be helpful.

# Llamaindex Q4

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

q2 = "How many Super Bowls have the Colts won?"
response = query_engine.query(q2)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# A faithfulness score of 1.0 means the RAG implementation has passed the evaluation. 
# The response has remained faithful to the contexts of the retrieved chunks. A score 
# of 0.0 implies it has not remained faithful to the contexts of the retrieved 
# information, and has provided halucinations or inaccurate information.
# Relevancy measures the relevancy of the response to the query using the retrieved 
# chunks, whereas faithfulness measures the relevancy of the response to the contexts 
# of the retrieved chunks.
# Yes, the scores changed between my two examples. The first example was relevant to 
# the documents and produced a response related to the query and faithful to the 
# retrieved contexts. The second question had no relevance to the provided documents. 
# So, the response was not relevant to the query or faithful to the retrieved contexts.
# The "LLM-as-a-judge" approach uses an appropriately trained llm to evaluate the RAG 
# implementation at a very fast rate. It evaluates based on the accuracy of natural 
# language rather than the numerical or categorical assessments of a simple accuracy 
# metric.
