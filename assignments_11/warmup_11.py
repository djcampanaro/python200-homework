# --- Prefect Orchestration ---

# Prefect Q1

# @task decorates a function with one objective that is part of the larger pipeline. 
# It is used the cache and retry the pipeline section as directed. @flow decorates 
# the main pipeline function that calls each of the tasks. You would not need to use 
# @task in the example of a pure, in-memory helper function as it does not require 
# connecting to an outside API. There is also no need to cache or retry since the 
# function is not calling out to a network.

# Prefect Q2

from prefect import task

@task(retries=3, retry_delay_seconds=30)
def call_api():
    pass

# Prefect Q3

# In the Prefect UI, I would click on the pipeline run, Then, I'd click on the red task 
# that failed and look at the Logs tab to find out what went wrong. I would expect to 
# find confirmation of the two completed tasks followed by a print out of what went wrong 
# during the transform task.

# --- Production Patterns ---

# Production Q1

# raise_for_status() is a loud failure that creates an exception that Prefect will catch 
# and mark the task as failed. 'if response.status_code != 200:' is a silent failure that 
# may pass corrupted and/or misleading data further down the pipeline. This may allow the 
# pipeline to complete successfully and can create a malformed result. 

# Production Q2

# In this scenario, upsert protects from a code-breaking error that would stop the request 
# immediately. If insert had been used, the first line of data would contain the primary 
# key of a line that already exists in the table as it is the exact same line of data being 
# inserted on the second try. Insert would catch this and cause an error as every primary 
# key must be unique. Upsert will instead update each line in place using each primary key 
# and then continue uploading the rows of data that were previously missed.

# Production Q3

from prefect.logging import get_run_logger

@task
def records_count(enrichment_records: list) -> None:
    logger = get_run_logger()
    logger.info(f"Upserted {len(enrichment_records)} enrichment records")

# Production Q4

# The incremental processing check in the transform task contributes to idempotency because it 
# separates records that have already been processed from those that have not. This way only the 
# records that still need processing are run through the ML and LLM steps of the pipeline. If we 
# were to run all 365 records every time, it would increase the overall cost due to redundant 
# calls to the LLM for pre-processed data. It slows down the pipeline as each call costs a small 
# amount of time that adds up, and it would replace the data that has already been upserted to 
# the table. If there was an issue with this particular run, it could replace good data with 
# something unintended.
