# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:54:20 2023

@author: tfvs06_1
"""
import os 
import pandas as pd
import numpy as np
from tensorflow.keras.layers import StringLookup

#%%
def get_dataset(dataset_folder):            # 將label以及csv檔案讀取位置製作成一個df
    tag_list  = os.listdir(dataset_folder)
    tag_n_path_df = []
    for tag in tag_list:
        csv_folder = os.path.join(dataset_folder, tag)
        csv_list = os.listdir(csv_folder)
        for csv_name in csv_list:
            csv_path = os.path.join(csv_folder, csv_name)
            tag_n_path_df.append((tag, csv_path))
    tag_n_path_df = pd.DataFrame(tag_n_path_df, columns=["label","csv_path"])
    return tag_n_path_df

def numericalize(label_n_path):
    label_processor = StringLookup(num_oov_indices= 0, vocabulary = np.unique(label_n_path["label"])) # 建立數字分類器
    labels = label_n_path["label"].values # 將label取出，目前為[Fall NotFall]
    numeric = label_processor(labels).numpy() # 丟入數字分類器，現為[0 1]
    label_n_path["label"] = numeric  # 放回原本位置
    return label_n_path

def load_CSV(tag_n_path_df):
    X = []
    for tag_n_path in tag_n_path_df.values:     
        data = pd.read_csv(tag_n_path[1])   # 讀取該位置的csv檔
        data = data.drop(data.columns[0], axis=1)   # 將無用訊息(frame_num)drop掉
        if len(data) < 18:                      # 如果不夠18幀則取最後1幀補至18幀長度
            last_row = data.iloc[-1]
            data = pd.concat([data, pd.DataFrame([last_row]*(18-len(data)))], ignore_index= True)
        else:                   # 超過則取最後18幀
            data = data[-18:]
        data = data.to_numpy()
        X.append(data)
    return np.array(X)

def random_split(numeric_label_n_path, ratio=0.8): # 將資料集打亂並拆成train及test兩個部分，默認比例為0.8
    indexes = np.random.permutation(numeric_label_n_path.shape[0]) # 根據資料筆數先創建一個nparray，內容為:0到該數不重複的隨機排序
    
    train_indexes = indexes[:int(numeric_label_n_path.shape[0] *ratio)] #[頭:0.8]
    test_indexes = indexes[int(numeric_label_n_path.shape[0] *ratio):]  #[0.8:底]
    
    train_tag_path = numeric_label_n_path.loc[train_indexes]
    test_tag_path = numeric_label_n_path.loc[test_indexes]
    
    train_X = load_CSV(train_tag_path)
    test_X = load_CSV(test_tag_path)
    
    train_y = np.expand_dims(train_tag_path["label"].to_numpy(), axis=1)    # 將y的shape修至與X對應
    test_y = np.expand_dims(test_tag_path["label"].to_numpy(), axis=1)
    
    return train_X, train_y, test_X, test_y

#%%

if __name__ == "__main__": 
#%%
    dataset_folder = "CSV_Dataset"   

#%%
    label_n_path = get_dataset(dataset_folder)
    # print(label_n_path)
    numeric_label_n_path = numericalize(label_n_path)
    # print(numeric_label_n_path)
    train_X, train_y, val_X, val_y, test_X, test_y = random_split(numeric_label_n_path)
    
    # print(test_X)
    # print(test_y)
