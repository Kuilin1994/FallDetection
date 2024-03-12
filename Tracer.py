# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:50:28 2023

@author: tfvs06_1
"""

import numpy as np

def check_size(prev_boxes_xyxy, boxes_xyxy):    # 此為炫技手法，有更簡單的寫法-> while迴圈
    if len(prev_boxes_xyxy) < len(boxes_xyxy):  # 如果上一幀物件少於這一幀
        return check_size(np.append(prev_boxes_xyxy, np.array([[None]*4]), axis=0), boxes_xyxy) # 將其插入[None,None,None,None]擴展至一樣長度
    else:
        return prev_boxes_xyxy
    
def compare(A, B):
    return list(set(A)-set(B))  #傳回A有B卻沒有的值 A:[0,1,2,3]、B:[2,3,4,5,6] 回傳:[0,1]

def duplicates(list_to_check):
    collections = {}
    for index, value in enumerate(list_to_check): # 製作一個dictionary，內容為紀錄重複的數字(id)及其index
        if f"{value}" not in collections:         # ex: id_list: [0,1,1,2,3,2,1,4,5] 回傳: {"1":[1,2,6], "2":[3,5]}        
            collections[f"{value}"] = []            
        collections[f"{value}"].append(index)
    # 僅保留有重複的部分    
    collections = {value: indexes for value, indexes in collections.items() if len(indexes) > 1 }
    return collections

def replace_same(list_to_rewrite, list_to_check_diff):              
    for element, indexes in duplicates(list_to_rewrite).items():
        # 留下第一個，剩餘改成None
        for index in indexes[1:]:
            list_to_rewrite[index] = None
    lost = compare(list_to_check_diff, list_to_rewrite) # 記錄遺失的id
    none_indexes = [index for index, value in enumerate(list_to_rewrite) if value is None]
    for i, index in enumerate(none_indexes): # 將遺失的id分配至None的位置
        list_to_rewrite[index] = lost[i]        
    return list_to_rewrite
    

    
#%%
class Tracer:    
    def __init__(self, YOLO_results, person_id=[], last_used_id=None, pre_boxes_xyxy=None):
        self.results = YOLO_results[0]
        self.keypoints_data = self.results.keypoints.data
        self.keypoints_xy = self.results.keypoints.xy.numpy()
        self.boxes =  self.results.boxes
        self.boxes_xyxy = self.boxes.xyxy.numpy()
        self.person_id = person_id
        self.prev_boxes_xyxy = pre_boxes_xyxy
        self.last_used_id = last_used_id
        
#%%    
    def trace(self):
        #Case0: 若前資訊盒列表為空，將現有資訊盒列表當作初始值
        if self.prev_boxes_xyxy is None:   #若前資訊盒列表為空，將現有資訊盒列表當作初始值
            self.prev_boxes_xyxy = self.boxes_xyxy
            self.person_id = list(range(len(self.boxes_xyxy))) #根據現有資訊盒的量賦予id
            if self.person_id != []:
                self.last_used_id = self.person_id[-1]
            else:
                self.last_used_id = -1
        else:
            # 記錄前資訊盒原本的長度
            prev_boxes_xyxy_length = len(self.prev_boxes_xyxy)
            # 將prev_boxes的大小補成與boxes_xyxy相同
            self.prev_boxes_xyxy = check_size(self.prev_boxes_xyxy, self.boxes_xyxy)
            # 計算方塊之間的差異矩陣，大小為i*j，i=j=N，N為資訊盒的量
            diff_matrix = []
            for i in range(len(self.boxes_xyxy)):          #現資訊盒有i個
                diff_matrix.append([])                     
                for j in range(len(self.prev_boxes_xyxy)): #前資訊盒有j個
                    diff = 0     #初始差異為0
                    for index in range(4):   #差異指標有4個:x1,y1,x2,y2
                        if self.prev_boxes_xyxy[j][index] is not None:
                            diff += abs(self.boxes_xyxy[i][index] - self.prev_boxes_xyxy[j][index])
                        else:
                            diff = None
                    diff_matrix[-1].append(diff) #填入差異
                    
            best_guess_id = [None] * len(self.boxes_xyxy)     #創造臨時id列表，長度與現資訊盒數相同，內容為空
            diff_matrix = np.array(diff_matrix)               #轉成np_array

#%%         Case1: 現資訊盒量(i)小於前資訊盒量(j),以i查詢j的id
            if len(self.boxes_xyxy) < len(self.prev_boxes_xyxy):
                # print(diff_matrix)
                for i in range(len(self.boxes_xyxy)):   
                    min_diff = 999999      
                    for j in range(len(self.prev_boxes_xyxy)):
                        check = diff_matrix[i][j]
                        if check < min_diff:
                            min_diff = check
                            best_guess_id[i] = self.person_id[j]
#%%
                # # 紀錄多個i都想拿同一個j，導致沒被拿而流失的j(包含多出的)
                # miss_id = compare(self.person_id, best_guess_id)
                # # 查看重複被拿取的j及拿取的i_index
                # if duplicates(best_guess_id):
                #     for id_, i_indexes in duplicates(best_guess_id).items():
                #         # 每一項重複的id留第一個，剩下改為None
                #         for i_index in i_indexes[1:]:
                #             best_guess_id[i_index] = None
                # # 將為None的i_index記錄下來            
                # none_indexes = [index for index, value in enumerate(best_guess_id) if value is None]
                # # 將流失的j補回
                # for index, id_ in zip(none_indexes, miss_id[:len(none_indexes)]):
                #     best_guess_id[index] = id_
#%%
                best_guess_id = replace_same(best_guess_id, self.person_id) # 原為上面的程式碼，已整理成function，留下註解是為了提供參考
                
                self.person_id = best_guess_id                
                self.prev_boxes_xyxy = self.boxes_xyxy
                
#%%         Case2: 現資訊盒量(i)大於等於前資訊盒量(j),以j分配i的id
            else:
                diff_matrix = diff_matrix.T
                # print(diff_matrix)
                which_i_to_give = []
                for j in range(prev_boxes_xyxy_length):
                    which_i_to_give.append(None)
                    min_diff = 999999
                    for i in range(len(self.boxes_xyxy)):
                        check = diff_matrix[j][i]
                        if check is None: 
                            pass
                        elif check < min_diff: 
                            min_diff = check
                            which_i_to_give[-1] = i
#%%
                # # 查看被重複分配的i_index及發配的j_index          
                # if duplicates(which_i_to_give):
                #     for i_indexes, j_indexes in duplicates(which_i_to_give).items():
                #         for j_index in j_indexes[1:]:
                #             which_i_to_give[j_index] = None 
                # # 將沒分配人的j_idex記錄下來
                # none_indexes = [j_index for j_index, i_index in enumerate(which_i_to_give) if i_index is None]
                # # 將沒被分配的i_index記錄下來
                # no_id_i = compare(range(len(self.boxes_xyxy)), which_i_to_give)
                # # print(no_id_i)
                # for j_index, i_index in zip(none_indexes, no_id_i[:len(none_indexes)]):
                #     which_i_to_give[j_index] = i_index
                # # print(which_i_to_give)
#%%                
                which_i_to_give = replace_same(which_i_to_give, range(len(self.boxes_xyxy)))  # 原為上面的程式碼，已整理成function，留下註解是為了提供參考           
                for j, i in enumerate(which_i_to_give):
                    best_guess_id[i] = self.person_id[j]
                    
                self.person_id = best_guess_id
                # 檢查沒被分配到的i再分配新id
                none_indexes = [index for index, value in enumerate(self.person_id) if value is None]
                if none_indexes:
                    to_fill = len(self.boxes.xyxy) - prev_boxes_xyxy_length
                    for i, extra_id in zip(none_indexes, range(self.last_used_id+1, self.last_used_id+to_fill+1)):
                        best_guess_id[i] = extra_id
                    self.last_used_id += to_fill
                        
                self.person_id = best_guess_id
                self.prev_boxes_xyxy = self.boxes_xyxy
#%%     回傳追蹤結果
        return self.person_id, self.last_used_id, self.prev_boxes_xyxy
    


