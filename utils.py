import pandas as pd


CLEANED_ILI_TALLY_FILE_NAME = r"I:\2023 ILI Pipe Tally Consolidation\All_ILIData.csv"

def read_cleaned_ILI_tally() -> pd.DataFrame:
    # Read the file
    data = pd.read_csv(CLEANED_ILI_TALLY_FILE_NAME, low_memory=False)
    # Output the number of rows
    print(f"Total rows = {len(data)}")
    print(f"Headers: {list(data)}")
    return data