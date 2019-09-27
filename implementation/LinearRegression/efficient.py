from .helper import *
import pandas as pd

def go():
    df = pd.read_csv(get_dataset_path(), header=0)
    print(df.head())