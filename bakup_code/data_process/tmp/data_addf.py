import pandas as pd
import time

train = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")
features = pd.read_csv("data/item_features.csv")

def merge_y(a, b):
    a = list(a)
    a.append(b)
    return a
test["item_id_list"] = test.apply(lambda x: merge_y(x["item_id"], x["y"]), axis=1)
train["item_id_list"] = train.apply(lambda x: merge_y(x["item_id"], x["y"]), axis=1)

train = train[['session_id', 'wf', 'item_id_list']]
test = test[['session_id', 'wf', 'item_id_list']]

def get_cat_feature(item_list, cat_id):
    f_list = []
    for item in item_list:
        f_v = features[(features["item_id"]==item) & (features["feature_category_id"]==cat_id)]["feature_value_id"].values
        if f_v.shape[0] > 0:
            f_list.append(f_v[0])
        else:
            f_list.append(-1)
    return f_list

for i in range(1,74):
    test[f"f{i}-list"] = -1
    train[f"f{i}-list"] = -1

for i in range(1,74):
    stime = time.time()
    test[f"f{i}-list"] = test["item_id_list"].apply(lambda x: get_cat_feature(x,i))
    train[f"f{i}-list"] = train["item_id_list"].apply(lambda x: get_cat_feature(x,i))
    print(f"feature {i} took {time.time() - stime} seconds")

test.to_parquet("data/test_addf.parquet")
train.to_parquet("data/train_addf.parquet")
