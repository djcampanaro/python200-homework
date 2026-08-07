from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

if load_dotenv():
    print("Successfully loaded api key")
    print('\n')


def get_completion(messages: list, model="gpt-4o-mini", **kwargs):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages, 
        **kwargs,
    )
    return response


# --- Completions API ---

# API Q1

client = OpenAI()

# messages = [{
#     "role": "user",
#     "content": "What is one thing that makes Python a good language for beginners?"
# }]
# response = get_completion(messages=messages)

# print('API response: ', response.choices[0].message.content)
# print('Model used: ', response.model)
# print('Tokens used: ', response.usage)
# print('\n')

# API Q2

# prompt = "Suggest a creative name for a data engineering consultancy."
# temperatures = [0, 0.7, 1.5]
# messages = [{
#     'role': 'user',
#     'content': prompt
# }]

# for t in temperatures:
#     response = get_completion(messages=messages, temperature=t)
#     print(f'Temperature: {t}\nResponse: {response.choices[0].message.content}')

# The first output gives one solution. The second output responds in a more 
# conversational manner. It lists off 10 ideas for the name of a data engineering 
# consultancy. The third output once again only puts out one solution, with no 
# other commentary. If I wanted a consistent answer I would use a temperature of 
# 0.0 as that will always provide the highest rated response.

# API Q3

# messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}]

# response = get_completion(messages=messages, n=3, temperature=1.0)

# for c in response.choices:
#     print(c.message.content)

# API Q4

# messages=[{"role": "user", "content": "Explain how neural networks work."}]
# response = get_completion(messages=messages, max_tokens=15)

# print(response.choices[0].message.content)

# The response was abbreviated and even cut off mid sentence. It is good to utilize 
# max_tokens in a real application to keep responses more succint and cut down on the 
# costs of token usage.

# --- System Messages and Persona ---

# System Q1

# messages = [
#     {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
#     {"role": "user", "content": "I don't understand what a list comprehension is."}
# ]

# response = get_completion(messages=messages)

# print(response.choices[0].message.content)

# messages_2 = [
#     {"role": "system", "content": "You are an impatient, busy Python tutor. You always explain things succinctly and are a bit technical."},
#     {"role": "user", "content": "I don't understand what a list comprehension is."}
# ]

# response = get_completion(messages=messages_2)

# print(response.choices[0].message.content)

# The first response was warm and thoroughly explained list comprehensions. It was 
# encouraging and helpful. The second response was curt, but still gave an example 
# of how list comprehensions work. It just doesn't go into any more detail, and it 
# feels like interacting with a machine.

# System Q2

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "My name is Jordan and I'm learning Python."},
#     {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
#     {"role": "user", "content": "Can you remind me what my name is?"}
# ]

# response = get_completion(messages=messages)

# print(response.choices[0].message.content)

# The model knows Jordan's name because the context of who Jordan is 
# is provided in the API call. Each API call is stateless, but if the 
# previous conversation is stored and resent, the model will have all 
# the necessary information to continue the conversation. If we sent 
# another API call without the previous conversation and asked the model 
# what Jordan's name is, it would have no context and not be able to 
# answer the question.

# --- Prompt Engineering ---

# Prompt Q1

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for r in range(len(reviews)):
    messages = [{"role": "user", "content": f"Classify the sentiment of this review as positive, negative, or mixed: {reviews[r]}"}]
    response = get_completion(messages=messages)
    print(f'Review {r + 1}:')
    print('Result: ', response.choices[0].message.content)

# Prompt Q2

for r in range(len(reviews)):
    messages = [{
        "role": "user", 
        "content": f"""
            Classify the sentiment of this review as positive, negative, or mixed: {reviews[r]} 
            Example:
            Review: "Fast shipping but the item arrived damaged."
            Sentiment: mixed
            """
    }]
    response = get_completion(messages=messages)
    print(f'Review {r + 1}:')
    print('Result: ', response.choices[0].message.content)

# Yes, adding an example did change the format and output of the response from Q1. 
# In Q1, the responses were full sentences. With the third review it explained its 
# reasoning for labeling it mixed. In Q2, the response mimics the example it was sent. 
# It simple sends "Sentiment: <sentiment>" and there is no explaination or reasoning 
# reported.

# Prompt Q3


