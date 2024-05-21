import pandas as pd
import numpy as np
import os
import time

data_path = "data/process_data"
valid_sessions = pd.read_csv(os.path.join(data_path, "train_sessions_new.csv"))
valid_purchases = pd.read_csv(os.path.join(data_path, "train_purchases_new.csv"))
valid_sessions['date'] = pd.to_datetime(valid_sessions["date"])
valid_wf = pd.read_parquet("data/train.parquet")[["session_id","wf"]]

features_table_pd = pd.read_parquet(os.path.join(data_path, "tmp/item_features_table.parquet"))

valid_sessions_feat = valid_sessions.merge(features_table_pd, on="item_id")
valid_sessions_feat.sort_values(["session_id", "date"], inplace=True)
valid_sessions_feat = valid_sessions_feat.drop(columns=["date"])

valid_purchases_feat = valid_purchases.merge(features_table_pd, on="item_id")
valid_purchases_feat.sort_values(["session_id"], inplace=True)
valid_purchases_feat = valid_purchases_feat.drop(columns=["date"])

stime = time.time()
valid_sessions_feat_list = valid_sessions_feat.groupby("session_id", as_index = False).agg({'item_id':lambda x: list(x), 'f_1': lambda x: list(x), 'f_2': lambda x: list(x), 'f_3': lambda x: list(x), 'f_4': lambda x: list(x), 'f_5': lambda x: list(x), 'f_6': lambda x: list(x), 'f_7': lambda x: list(x), 'f_8': lambda x: list(x), 'f_9': lambda x: list(x), 'f_10': lambda x: list(x), 'f_11': lambda x: list(x), 'f_12': lambda x: list(x), 'f_13': lambda x: list(x), 'f_14': lambda x: list(x), 'f_15': lambda x: list(x), 'f_16': lambda x: list(x), 'f_17': lambda x: list(x), 'f_18': lambda x: list(x), 'f_19': lambda x: list(x), 'f_20': lambda x: list(x), 'f_21': lambda x: list(x), 'f_22': lambda x: list(x), 'f_23': lambda x: list(x), 'f_24': lambda x: list(x), 'f_25': lambda x: list(x), 'f_26': lambda x: list(x), 'f_27': lambda x: list(x), 'f_28': lambda x: list(x), 'f_29': lambda x: list(x), 'f_30': lambda x: list(x), 'f_31': lambda x: list(x), 'f_32': lambda x: list(x), 'f_33': lambda x: list(x), 'f_34': lambda x: list(x), 'f_35': lambda x: list(x), 'f_36': lambda x: list(x), 'f_37': lambda x: list(x), 'f_38': lambda x: list(x), 'f_39': lambda x: list(x), 'f_40': lambda x: list(x), 'f_41': lambda x: list(x), 'f_42': lambda x: list(x), 'f_43': lambda x: list(x), 'f_44': lambda x: list(x), 'f_45': lambda x: list(x), 'f_46': lambda x: list(x), 'f_47': lambda x: list(x), 'f_48': lambda x: list(x), 'f_49': lambda x: list(x), 'f_50': lambda x: list(x), 'f_51': lambda x: list(x), 'f_52': lambda x: list(x), 'f_53': lambda x: list(x), 'f_54': lambda x: list(x), 'f_55': lambda x: list(x), 'f_56': lambda x: list(x), 'f_57': lambda x: list(x), 'f_58': lambda x: list(x), 'f_59': lambda x: list(x), 'f_60': lambda x: list(x), 'f_61': lambda x: list(x), 'f_62': lambda x: list(x), 'f_63': lambda x: list(x), 'f_64': lambda x: list(x), 'f_65': lambda x: list(x), 'f_66': lambda x: list(x), 'f_67': lambda x: list(x), 'f_68': lambda x: list(x), 'f_69': lambda x: list(x), 'f_70': lambda x: list(x), 'f_71': lambda x: list(x), 'f_72': lambda x: list(x), 'f_73': lambda x: list(x)})
valid_sessions_feat_list.to_parquet(os.path.join(data_path, "tmp/train_sessions_feat_list.parquet"))
valid_sessions_feat_list = pd.read_parquet(os.path.join(data_path, "tmp/train_sessions_feat_list.parquet"))
print(f"group took {time.time()-stime} seconds")


valid_purchases_feat.columns = ["session_id", "item_id_y"] + [f"f_{i}_y" for i in range(1,74)]
valid_sessions_feat_list=valid_sessions_feat_list.merge(valid_purchases_feat,on="session_id", how="left")

stime = time.time()
def merge_y(a, b):
    a = list(a)
    a.append(b)
    return a
valid_sessions_feat_list["item_id-list"] = valid_sessions_feat_list.apply(lambda x: merge_y(x["item_id"], x["item_id_y"]), axis=1)

for i in range(1,74):
    valid_sessions_feat_list[f"f_{i}-list"] = valid_sessions_feat_list.apply(lambda x: merge_y(x[f"f_{i}"], x[f"f_{i}_y"]), axis=1)
print(f"merge y took {time.time()-stime} seconds")

valid_sessions_feat_list = valid_sessions_feat_list[["session_id", "item_id-list"] + [f"f_{i}-list" for i in range(1,74)]]
valid_sessions_feat_list = valid_sessions_feat_list.merge(valid_wf, on="session_id", how="left")
valid_sessions_feat_list.to_parquet("data_trans/train.parquet")
