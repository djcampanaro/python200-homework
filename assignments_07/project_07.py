import os
import pandas as pd

from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import tool

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
api_key = os.getenv("OPENAI_API_KEY")

# Task 1: Define Your Tools

df = None
error_df = {'error': 'There is no dataframe loaded'}

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Uses:
        DATA_PATH to find path to csv file. Creates the csv file if not found.
    Saves:
        Saves a pandas dataframe in a previously created global variable named df.
    Returns:
        A dictionary defining the shape and columns of the created dataframe.
    """
    global df

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        df_happiness = pd.DataFrame()
        numeric_columns = ['happiness_score','gdp_per_capita','social_support','healthy_life_expectancy','freedom_to_make_life_choices','generosity','perceptions_of_corruption']
        num_columns = numeric_columns.copy()
        num_columns.append('ladder_score')
        happiness_project_path = '../python-200/assignments/resources/happiness_project/world_happiness_20'
        for year_abrev in range(15, 25):
            year = f'20{year_abrev}'
            data_path = f'{happiness_project_path}{year_abrev}.csv'
            add_df = pd.read_csv(data_path, sep=';')
            add_df = add_df.sort_values(by='Ranking')
            column_names = list(add_df.columns)
            col_renames = {}
            for col in column_names:
                new_name = col.lower().replace(' ','_')
                col_renames[col] = new_name
                if new_name in num_columns:
                    try:
                        add_df[col] = add_df[col].apply(lambda x: x.replace(',', '.') if ',' in x else x).astype(float)
                    except TypeError:
                        add_df[col] = add_df[col].astype(float)
            col_renames['Ladder score'] = 'happiness_score'
            add_df = add_df.rename(columns=col_renames)
            add_df['year'] = year
            df_happiness = pd.concat([df_happiness, add_df])
            df_happiness = df_happiness.reset_index(drop=True)
        df_happiness.to_csv(DATA_PATH, index=False)
        df = df_happiness

    df_dict = {"shape": df.shape, "columns": df.columns}

    return df_dict

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """

    if df == None:
        return error_df
    
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: First column name to compare
        col2: Second column name to compare

    Returns: 
        A dict containing keys 'col1', 'col2', 'pearson_r', and 'p_value' along with their corresponding values or an error dict.
    """

    if df == None:
            return error_df
    
    if col1 not in df.columns:
        return {"error": f"'{col1}' is not a column. Options: {df.columns.tolist()}"}
    elif col2 not in df.columns:
        return {"error": f"'{col2}' is not a column. Options: {df.columns.tolist()}"}
    
    r, p = pearsonr(df[col1], df[col2])
    correlation = {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(r, 4),
        "p_value": round(p, 4)
    }

    return correlation

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    The year range is 2015-2024. 

    Args:
        column: the category of values to sort
        year: desired year to search for values 2015-2024
        n: the number of coutries/ values to return.

    Returns:
        A list of dictionaries containing keys 'country' and the selected column's name for the top n countries after sort
    """

    if df == None:
            return error_df
    
    try:
        if len(year) < 4:
            year = int('20' + year)
    except TypeError:
        if 14 < year < 25:
            year += 2000

    if not column or not year:
        return {"error": f"Missing a required variable. Column options: {df.columns.tolist()}, Year options: 2015-2024"}
    elif column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    elif year < 2015 or year > 2024:
        return {"error": f"'{year}' is not in range. Options: 2015-2024"}
    
    countries_sorted = df[df['year'] == year][['country', column]].sort_values(by=column, ascending=False).reset_index()
    if n > len(countries_sorted):
        return {'error': 'The requested number exceeds the number of countries with data that year.'}
    top_n_countries = []
    for i in range(n):
        top_n_countries.append({'country': countries_sorted.loc[i, 'country'], column: countries_sorted.loc[i, column]})

    return top_n_countries

@tool
def get_happiness_dataframe():
    """
    Takes the dataframe that is defined in this module and returns it to be used for parsing.

    Returns:
        Dataframe from current module 
    """
    return df

# Task 2: Build the Agent

from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries, get_happiness_dataframe],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# Task 3: Run Guided Queries

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)

# Task 4: Your Own Questions

# My query 1
my_query_1 = "Rank the top 5 countries for social_support over all the years provided."   # replace with your question
response_1 = agent.run(my_query_1, reset=False)
print(response_1)
# Comment: Did this trigger tool use, code generation, or both?
# This query triggered both tool use and code generation. The agent used the tool 
# get_top_n_coutries to gather a list of the top 5 countries for each year. It then 
# wrote its own code to find which countries had the most occurences and give the 
# answer of the top 5 over all years provided.

# My query 2
my_query_2 = "Plot the correlation between perceptions_of_corruption and healthy life expectancy for all countries in 2024. Save the plot to outputs/happiness_by_region.png."   # replace with your question
response_2 = agent.run(my_query_2, reset=False)
print(response_2)
# Comment: Did this trigger tool use, code generation, or both?
# This query triggered code generation. I had expected it to trigger tool use as well, 
# but the mention of correlation in the query may not have been enough to get the agent 
# to use the correlation tool. I left the plot description ambiguous to see what the agent 
# would decide to use. It created a scatter plot of the two columns and added a linear 
# regression line.

# Task 5: Reflection

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#
# The agent communicated the significance of the correlation as a key/value pair in its 
# response. Yes, it used the p-value correctly and surmised there is a statistical 
# significance between the two columns. I'm not certain which threshold is applied in this 
# situation. The agent passed the names of the two columns to the tool, which called pearsonr 
# to get the pearson r and p-value.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
# I was surprised when I first ran the module how long it took the agent to provide an answer 
# to the first query. There were multiple tries involved with the agent making an assumption 
# about what data it would receive from the first function. Despite the description stating 
# it would return the values the query asked for, the agent assumed it would receive the entire 
# dataframe and compute the shape and columns itself. This may be an issue with the description 
# or possibly the title of the function.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
# An additional tool that would make this agent meaningfully more useful is one that does 
# hypothesis testing. This would allow the user to compare data over regions or years to 
# find if there could be causation to a hypothesis they have.

