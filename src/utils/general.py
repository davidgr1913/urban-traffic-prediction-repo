import re

def to_snake_case(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '_')).lower().rstrip('_') for col in df.columns]
    return df