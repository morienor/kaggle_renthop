import pandas as pd
import numpy as numpy
from sklearn.model_selection import StratifiedKFold
import pickle

def IO_full():
	train_p, test_p = "downloads/train.json", "downloads/test.json"
	train=pd.read_json(train_p)
	test=pd.read_json(test_p)
	train["set"]='train'
	test["set"]='test'
	df_full = pd.concat([train, test])
	df_full = df_full.reset_index(drop=True) # Because stratkfold resets index
	df_train = df_full[df_full["set"] == "train"]
	df_test = df_full[df_full["set"] == "test"]
	return df_full, df_train, df_test

def IO_SkF(	):
	train_path = "skf/fold_1_train_index"
	valid_path = "skf/fold_1_test_index"
	with open(train_path, "rb") as f:
		train_index = pickle.load(f)
	with open(valid_path, "rb") as f:
		valid_index = pickle.load(f)
	return train_index, valid_index

def make_folds(df_train, df_test):
	col_feats = [col for col in df_train if col != "interest_level"]
	X = df_train[col_feats]
	target_num_map={"high":0, "medium":1, "low":2}
	y = df_train["interest_level"].apply(lambda x: target_num_map[x])	
	skf = StratifiedKFold(n_splits=5)
	print(count_1, count_2, df_train.shape[0], df_test.shape[0])
	with open("skf/fold_%s_train_index" % count, "wb") as f:
		pickle.dump(train_index, f, pickle.HIGHEST_PROTOCOL)
	with open("skf/fold_%s_test_index" % count, "wb") as f:
		pickle.dump(test_index, f, pickle.HIGHEST_PROTOCOL)




def make_simple_features(data):
	data["created"]=pd.to_datetime(data["created"])
	data["created_month"]=data["created"].dt.month
	data["created_day"]=data["created"].dt.day
	data["created_hour"]=data["created"].dt.hour
	data["num_photos"]=data["photos"].apply(len)
	data["num_features"]=data["features"].apply(len)
	data["num_description_words"] = data["description"].apply(lambda x: len(x.split(" ")))
	return data



