import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Task 1: Define Your Tools

df = None

# @tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    Saves a pandas dataframe in df.
    Uses:
        DATA_PATH to find path to csv file. Creates the csv file if not found.
    Returns:
        A dict with the shape and columns of the dataframe.
    """
    global df

    if not DATA_PATH:
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

    df = pd.read_csv(DATA_PATH)
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
    if col1 not in df.columns:
        return {"error": f"'{col1}' is not a column. Options: {df.columns.tolist()}"}
    elif col2 not in df.columns:
        return {"error": f"'{col2}' is not a column. Options: {df.columns.tolist()}"}
    
    col1 = df[col1]
    col2 = df[col2]
    
    r, p = pearsonr(col1, col2)
    correlation = {
        "col1": col1,
        "col2": col2,
        "person_r": round(r, 4),
        "p_value": round(p, 4)
    }

    return correlation

# @tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year between 2015-2024.
    ...
    """
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
    top_n_countries = []
    for i in range(n):
        top_n_countries.append({'country': countries_sorted.loc[i, 'country'], column: countries_sorted.loc[i, column]})

    return top_n_countries


