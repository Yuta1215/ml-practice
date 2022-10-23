import pandas as pd
import pickle
from sklearn import tree


PATH = '/Users/yuta/Development/ml-practice/src/main'


if __name__ == '__main__':
    df = pd.read_csv(f"{PATH}/iris.csv")
    t = df['種類'].unique()