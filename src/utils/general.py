import re

def to_snake_case(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col).lower().replace(' ', '_') for col in df.columns]
    return df