# --- ML vs. LLM in Pipelines ---

# ML/LLM Q1

# In this week's pipeline the ML classifier produces a binary answer (0 or 1) to signal days that are good for running. 
# Alongside that, it gives a confidence rating for how correct the answer is. The LLM is providing written text to make the 
# ML's response easier to digest by a human. If we were to switch their roles, the LLM may not provide the binary answer we need. 
# Instead, it would likely provide a longer response that would not work with our schema. The ML classifier wouldn't know what to 
# do with an LLM query and would error immediately. 

# ML/LLM Q2

# Converting a date string is best handled by deterministic code.
# For classifying a job posting based on freeform text, I would use an LLM.
# In predicting customer churn given 15 numeric features and a labeled training dataset I would employ a trained ML model.
# To normalize inconsistent city names, I would utilize an LLM.
# Summing a column of revenue figures works well with deterministic code.

# ML/LLM Q3

# Incremental advances the pipeline forward without redoing work. It only puts data through the classifier that has yet to be 
# enriched. This keeps costs lower as you won't need to rerun 360 rows everytime the scripts ran. If something were to go wrong 
# with the last step, you would want to be able to redo the llm enrichment without running the data through the classifier again.

# --- Prompt Design ---

# Prompt Q1

# "You are writing a two-sentence running recommendation for a daily weather summary app. You will receive weather conditions 
# for a single day and a machine learning prediction about whether the day is good for running. Write exactly two sentences — 
# direct, practical, and specific to the conditions. In the first sentence,  simply state the prediction in plain language. In 
# the second sentence, explain the reasoning for the prediction. Do not use bullet points, headers, or phrases like 'Based on the 
# data'."

# In the validation function, we would need to increase the limit for the number of sentences by one. We should also increase the 
# max tokens, possibly double them, to allow some headroom for the second sentence.

# Prompt Q2

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call_with_retry(client, messages, max_retries=3):
    while tries > max_retries:
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return res
        except Exception as e:
            print(e)
            tries -= 1
            time.sleep(2)

# You would use this in a production pipeline when you are calling into a network that may fail intermittently. In this case you 
# would give a call to the AI a few tries in case an exception arose. This would be useful for chatbots or programs digesting and 
# enriching large amounts of data.
