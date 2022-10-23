import pandas as pd
import pickle
from sklearn import tree


PATH = '/Users/yuta/Development/ml-practice/src/main'


if __name__ == '__main__':
    df = pd.read_csv(f"{PATH}/sample.csv")
    xcol = ['身長', '体重', '年代']
    x = df[xcol]
    t = df['派閥']
    print(x)
    print(t)
    # model = tree.DecisionTreeClassifier(random_state=0)
    # with open(f"{PATH}/sample.pkl", 'wb') as f:
    #     pickle.dump(model, f)
    with open(f"{PATH}/sample.pkl", 'rb') as f:
        model2 = pickle.load(f)
    # model.fit(x, t)
    model2.fit(x, t)
    taro = [170, 70, 20]
    taro_2 = [172, 60, 30]
    new_data = [taro, taro_2]
    # result = model.predict(new_data)
    result = model2.predict(new_data)
    print(result)
    # score = model.score(new_data, ['きのこ', 'たけのこ'])
    score = model2.score(new_data, ['きのこ', 'たけのこ'])
    print(score)