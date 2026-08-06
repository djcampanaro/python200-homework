from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint

if load_dotenv():
    print("Successfully loaded api key")
    print('\n')

# --- Completions API ---

# API Q1

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{
        "role": "user",
        "content": "What is one thing that makes Python a good language for beginners?"
    }],
)

print('API response: ', response.choices[0].message.content)
print('Model used: ', response.model)
print('Tokens used: ', response.usage)
print('\n')

# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for t in temperatures:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages = [{
            'role': 'user',
            'content': prompt
        }],
        temperature=t
    )
    print(f'Temperature: {t}\nResponse: {response.choices[0].message.content}')

