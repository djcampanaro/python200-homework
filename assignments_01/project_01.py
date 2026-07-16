import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from prefect import flow, task
from prefect.logging import get_run_logger
from scipy.stats import ttest_ind, pearsonr


# Task 1: Load Multiple Years Of Data
@task(retries=3, retry_delay_seconds=2)
def load_multiple_years_of_data(numeric_columns: list, db_path: str):
    logger = get_run_logger()
    # Create list of urls for datasets
    url_prefix = 'https://raw.githubusercontent.com/Code-the-Dream-School/python-200-v1/refs/heads/main/assignments/resources/happiness_project/world_happiness_20'
    years_list = [15,25]
    urls = [f'{url_prefix}{year}.csv' for year in range(years_list[0], years_list[1])]

    df_happiness = pd.DataFrame()
    num_columns = numeric_columns.copy()
    num_columns.append('ladder_score')
    for url in urls:
        year = url.split('ss_')[-1].split('.csv')[0]
        add_df = pd.read_csv(url, sep=';')
        add_df = add_df.sort_values(by='Ranking')
        column_names = list(add_df.columns)
        col_renames = {}
        for col in column_names:
            new_name = col.lower().replace(' ','_')
            col_renames[col] = new_name
            if new_name in num_columns:
                logger.info(new_name)
                try:
                    add_df[col] = add_df[col].apply(lambda x: x.replace(',', '.') if ',' in x else x).astype(float)
                except TypeError:
                    add_df[col] = add_df[col].astype(float)
                logger.info('alright')
        col_renames['Ladder score'] = 'happiness_score'
        add_df = add_df.rename(columns=col_renames)
        add_df['year'] = year
        df_happiness = pd.concat([df_happiness, add_df])
        df_happiness = df_happiness.reset_index(drop=True)
    df_happiness.info()
    df_happiness.to_csv(db_path, index=False)
    logger.info("dataframe successfully created")


# Task 2: Descriptive Statistics
@task
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    happiness_mean = df['happiness_score'].mean()
    happiness_median = df['happiness_score'].median()
    happiness_std = df['happiness_score'].std()
    logger.info(f'Happiness Score Mean: {happiness_mean}')
    logger.info(f'Happiness Score Median: {happiness_median}')
    logger.info(f'Happiness Score Standard Deviation: {happiness_std}')
    happiness_by_year_and_region = df.groupby(['year', 'regional_indicator']).agg({'happiness_score': ['mean']})
    logger.info(f'Happiness Score Grouped By Year and Region: {happiness_by_year_and_region}')
    return df


# Task 3: Visual Exploration
@task
def visual_exploration(df, numeric_columns):
    logger = get_run_logger()
    plt.hist(df['happiness_score'], bins=10, color='blue')
    plt.savefig('assignments_01/outputs/happiness_histogram.png', dpi=300)
    logger.info('happiness histogram plot saved.')
    plt.clf()

    ax = df.boxplot(by="year", column="happiness_score")
    plt.title("Happiness Scores by Year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig('assignments_01/outputs/happiness_by_year.png', dpi=300)
    logger.info('happiness by year plot saved.')
    plt.clf()

    scatter_data = df[['year', 'gdp_per_capita', 'happiness_score']]
    sns.scatterplot(scatter_data, x='gdp_per_capita', y="happiness_score", hue='year')
    plt.savefig('assignments_01/outputs/gdp_vs_happiness.png', dpi=300)
    logger.info('gdp vs happiness plot saved.')
    plt.clf()

    heatmap = df[numeric_columns]
    happiness_correlation = heatmap.corr(numeric_only=True)
    sns.heatmap(happiness_correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.savefig('assignments_01/outputs/correlation_heatmap.png', dpi=300)
    logger.info('correlation heatmap plot saved.')
    plt.close()


# Task 4: Hypothesis Testing
@task
def hypothesis_testing(df: pd.DataFrame):
    logger = get_run_logger()
    year_a = df[df['year'] == 2019]['happiness_score']
    year_b = df[df['year'] == 2020]['happiness_score']

    t_stat, p_val = ttest_ind(year_a, year_b)
    logger.info("T-test result: t=%.2f, p=%.4f", t_stat, p_val)
    logger.info("2019 mean: %.1f, 2020 mean: %.1f", year_a.mean(), year_b.mean())
    logger.info('The p-value is more than 50% chance of being luck, and the means are similar. Therefore the difference in results are likely up to chance.')

    region_a = df[df['regional_indicator'] == 'South Asia']['happiness_score']
    region_b = df[df['regional_indicator'] == 'East Asia']['happiness_score']

    t_stat, p_val = ttest_ind(region_a, region_b)
    logger.info("T-test result: t=%.2f, p=%.10f", t_stat, p_val)
    logger.info("South Asia mean: %.1f, East Asia mean: %.1f", region_a.mean(), region_b.mean())


# Task 5: Correlation and Comparisons
@task
def correlation_and_comparisons(df, numeric_columns) -> str:
    logger = get_run_logger()
    significant_comparisons = []
    significant_comparisons_adjusted = []
    adjusted_alpha = .05 / (len(numeric_columns) - 1)
    logger.info(adjusted_alpha)
    pvals = {}
    for col in range(1, len(numeric_columns)):
        r, p = pearsonr(df[numeric_columns[0]], df[numeric_columns[col]])
        logger.info(f"Happiness Score vs. {numeric_columns[col].replace('_', ' ').title()}")
        logger.info(f"Pearson Result: r={r}, p={p}")
        if p < .05:
            significant_comparisons.append(numeric_columns[col])
        if p < adjusted_alpha:
            significant_comparisons_adjusted.append(numeric_columns[col])
        pvals[numeric_columns[col]] = p
    pval_min = 1.0
    pval_min_key = ''
    for key, value in pvals.items():
        if value < pval_min:
            pval_min = value
            pval_min_key = key
    logger.info(f"Significant comparisons before adjustment: {', '.join(significant_comparisons)}")
    logger.info(f"Significant comparisons after adjustment: {', '.join(significant_comparisons_adjusted)}")
    return pval_min_key.replace('_', ' ').title()


# Task 6: Summary Report
@task
def summary_report(df, pval_minimum):
    logger = get_run_logger()
    countries_unique = df['country'].unique()
    num_countries = len(countries_unique)
    region_mean = df.groupby('regional_indicator')['happiness_score'].mean()
    region_mean = region_mean.sort_values(ascending=False)
    top_3 = ', '.join(region_mean.index[:3])
    bottom_3 = ', '.join(region_mean.index[-3:])
    logger.info('The data shows a higher mean of happiness in more developed regions, and areas with greater focus on human rights.')
    logger.info(f'The total number of countries in the dataset is {num_countries}')
    logger.info(f'The top three regions in mean happiness score are {top_3}')
    logger.info(f'The bottom three regions in mean happiness score are {bottom_3}')
    logger.info('The mean happiness score in 2020, after the start of the pandemic, was not different enough from the previous year to rule out chance as an explanation.')
    logger.info(f'The catagory most closely related to happiness score is {pval_minimum}')


@flow
def happiness_pipeline():
    numeric_columns = ['happiness_score','gdp_per_capita','social_support','healthy_life_expectancy','freedom_to_make_life_choices','generosity','perceptions_of_corruption']
    db_path = 'assignments_01/outputs/merged_happiness.csv'
    if os.path.exists(db_path):
        answer = input("The database exists.  Do you want to recreate it (y/n)?")
        if answer.lower() == 'y':
            os.remove(db_path)
            df = load_multiple_years_of_data(numeric_columns, db_path)
    else:
        df = load_multiple_years_of_data(numeric_columns, db_path)
    df = pd.read_csv(db_path)
    descriptive_stats(df)
    visual_exploration(df, numeric_columns)
    hypothesis_testing(df)
    pval_minimum = correlation_and_comparisons(df, numeric_columns)
    summary_report(df, pval_minimum)


if __name__ == "__main__":
    happiness_pipeline()
