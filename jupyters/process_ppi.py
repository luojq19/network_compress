# In[]
import pandas as pd
import os

def filter_ppi(ppi_path):
    df = pd.read_csv(ppi_path, sep=' ')
    # filter rows with experiments > 0 and database > 0
    df_filtered = df[(df['experiments'] > 0) & (df['database'] > 0)].reset_index(drop=True)
    return df_filtered

df_filtered = filter_ppi('../data/STRING/9606.protein.physical.links.full.v12.0.txt')
df_filtered
# %%
