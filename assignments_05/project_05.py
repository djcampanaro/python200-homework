import json

# Task 1: Setup and System Prompt

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

prompt = 'You are a job application coach. You are helping a user who ' \
'is changing careers and needs to update their resume and application ' \
'materials using their previously earned skills. These skills should be ' \
'updated in a way that applies to the new career path. Always use skills ' \
'listed by the user, and never make up qualifications. Stay focused on ' \
'job application materials. Always remind the user to review and edit all ' \
'outputs before submitting their application anywhere. Acknowledge that ' \
'you may not know much about the particular industry norms, and that the ' \
'user should use their best judgment.'

messages = [{"role": "system", "content": prompt}]

response = get_completion(messages=messages)

# One deliberate choice was to ask the model to not make up qualifications for 
# the user. I wanted to be sure that the model stuck to the skills and experience 
# that is listed by the user and didn't overelaborate their abilities.

# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list, with no delimiters. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result

    response = get_completion(messages=messages)

    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        print(response)

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

# bullets_rewrite = rewrite_bullets(bullets)
# for r in bullets_rewrite:
#     print('Original:', r['original'])
#     print('Improved:', r['improved'])

# These bullets are weak because they are very general and do not exemplify 
# how the action affected the company with positive results. The improved 
# results create stronger action words and identify direct effects of how 
# they helped improve the company.

# Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Example 3:
    Role: Data Technician
    Background: Several years experience in data entry and analytics with Python and SQL.
    Opening: As a highly organized and analytical individual with a proven track record of success in 
    data entry and analysis, I am confident that I can make a positive contribution to your team.
    I have a strong background in data entry and analysis, with experience in a variety of software 
    applications, including Microsoft Excel, SPSS, and Access. I am also knowledgeable in the use of 
    various statistical techniques, such as regression analysis and ANOVA. I am highly skilled in data 
    manipulation and have a keen eye for detail, ensuring accuracy in all data entry and analysis tasks.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    response = get_completion(messages=messages)
    return response

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

cover_letter = generate_cover_letter(job_title, background)
print(cover_letter)

# I used these examples as two were provided in the assignment. The third is more in line with something 
# I might want to do and I think the opening gives a strong sense of the candidates experience and ability 
# to perform the position.

# The few-shot pattern helps control the tone and it pushes the model toward the desired format and logic.

# Task 4: Moderation Check

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print('This message is not acceptable in its current state. Please rephrase your message.')
        return False
    return True

is_safe("I want to kill my neighbor.")
is_safe("I want to water my neighbor's plants.")

# Task 5: The Chatbot Loop

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            bullets_rewrite = rewrite_bullets(raw_bullets)
            if bullets_rewrite:
                for r in bullets_rewrite:
                    print('Original:', r['original'])
                    print('Improved:', r['improved'])

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter = generate_cover_letter(job_title, background)
            print(cover_letter)

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            response = get_completion(messages=messages)
            # - Print the reply
            print('Job Application Coach:', response)
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": response})
            pass


if __name__ == "__main__":
    run_chatbot()

# --- Option A ---

# 2. If the user were submit the model's response to an employer without reviewing it, 
# they could end up sending incorrect or incomplete information. If the model were to 
# return something that either did not fit the job or explicitly was outsite the bounds 
# of applicant's cover letter/resume, it would look quite bad to the prospective employer.

# 3. If this were deployed professionally, I would add a UI warning/disclaimer regarding 
# the user's need to review all responses and that the model may not fit their specific 
# industry. I would also make sure that there was a moderation guardrail in place to ensure 
# that the chat is solely based around the job application help and doesn't veer into 
# controvertial topics.
