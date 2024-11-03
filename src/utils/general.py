import re

def to_snake_case(df):
    """
    Converts the column names of a DataFrame to snake_case.

    This function takes a DataFrame and modifies its column names by:
    - Replacing spaces with underscores.
    - Removing any characters that are not alphanumeric or underscores.
    - Converting all characters to lowercase.
    - Removing trailing underscores.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose column names are to be converted.

    Returns:
    pandas.DataFrame: The DataFrame with modified column names.
    """
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '_')).lower().rstrip('_') for col in df.columns]
    return df