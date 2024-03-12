# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 23:56:36 2023

@author: bear7
"""
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np
import pandas as pd

from Tracer import Tracer
import LSTM



#%%
#座標相對化+正規化
def coordinate_normalize(keypoints_xy,boxes_xyxy):
    keypoints_xy = keypoints_xy
    
    boxes_xy = boxes_xyxy[:,:2] #左上角
    boxes_xy = np.expand_dims(boxes_xy, axis=1)
    
    delta_x = (boxes_xyxy[:,2]-boxes_xyxy[:,0]) #寬
    delta_y = (boxes_xyxy[:,3]-boxes_xyxy[:,1]) #高
    boxes_wh = np.vstack((delta_x, delta_y)).T
    boxes_wh = np.expand_dims(boxes_wh, axis=1)
    # 對座標進行標準化
    normalized_keypoints_xy = (keypoints_xy - boxes_xy) / boxes_wh 
    normalized_keypoints_xy = list(map(lambda x: pd.DataFrame(x.reshape(1,-1), columns=keypoint_coordinate_list), normalized_keypoints_xy))
             
    return normalized_keypoints_xy

#畫骨骼圖(註解部分為切割)
def draw_bone_and_segment(capture, boxes_xyxy_int, keypoints_data):
    capture_hw = capture.shape[:2]
    # 創建標註器
    skeleton = Annotator(capture)
    skeleton.kpts(keypoints_data, capture_hw)
    
    return skeleton.result()

#使用LSTM偵測
def detect_fall(person_df):
    person_df = person_df.values
    person_df = np.expand_dims(person, axis=0)
    prediction = LSTM_model.predict(person, verbose=False)
    if prediction[0][0] < 0.5:  #以0.5為閾值，0為跌倒，1為非跌倒
        return True
    else:
        return False

#%%

model = YOLO('yolov8n-pose.pt')


keypoint_coordinate_list = ["nose_X","nose_Y",
                            "left_eye_X","left_eye_Y",
                            "right_eye_X","right_eye_Y",
                            "left_ear_X","left_ear_Y",
                            "right_ear_X","right_ear_Y",
                            "left_shoulder_X","left_shoulder_Y",
                            "right_shoulder_X","right_shoulder_Y",
                            "left_elbow_X","left_elbow_Y",
                            "right_elbow_X","right_elbow_Y",
                            "left_wrist_X","left_wrist_Y",
                            "right_wrist_X","right_wrist_Y",
                            "left_hip_X","left_hip_Y",
                            "right_hip_X","right_hip_Y",
                            "left_knee_X","left_knee_Y",
                            "right_knee_X","right_knee_Y",
                            "left_ankle_X","left_ankle_Y",
                            "right_ankle_X","right_ankle_Y"]

#%% 定義變數初值
person_id = []
last_used_id = None
pre_boxes_xyxy=None
df_keypoints = []

LSTM_model = LSTM.load()

video_device = cv2.VideoCapture("video.mp4") # 可換成影片位置或是填入0使用webcam
video_device.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#%%
try:
    while True:
        success, capture = video_device.read()
        if success:
            results = model(capture, verbose=False)
            tracer = Tracer(results, person_id, last_used_id, pre_boxes_xyxy)
            person_id, last_used_id ,pre_boxes_xyxy = tracer.trace() 
            normalized_keypoints_xy = coordinate_normalize(tracer.keypoints_xy, tracer.boxes_xyxy)    
            keypoints_data_list  = tracer.keypoints_data
            
            keypoints_of_each = [None]*(last_used_id+1) #創建一個長度為物件數的空串列
            boxes_xyxy_list = []
            for person in tracer.boxes_xyxy:
                boxes_xyxy_list.append(list(map(lambda x: int(x), person))) #將其整數化後放入串列
            for index, id_ in enumerate(person_id):
                capture = draw_bone_and_segment(capture, boxes_xyxy_list[index], keypoints_data_list[index]) #畫骨骼
                keypoints_of_each[id_] =  normalized_keypoints_xy[index]        # 照trace後得到的id順序排列  
                
                
            # 將每幀資料以向下擴展方式合併    
            while len(df_keypoints) < len(keypoints_of_each): # 若發現合併時物件數多於以前
                df_keypoints.append(None)                     # 新增空欄位
                
            index_to_pop = []
            for id_, data in enumerate(keypoints_of_each):  
                if data is not None: 
                    if df_keypoints[id_] is None:   # 若要合併的位置為空(id為新的新物件)
                        df_keypoints[id_] = pd.concat([data]*18) # 直接複製其值並擴展到LSTM可用長度當初值(我的LSTM一次看18幀)
                    else: 
                        df_keypoints[id_] = pd.concat([df_keypoints[id_].iloc[1:], data]) # 丟棄原df第一row後往下新增
                else: # 若發現有物件消失內容為空
                    index_to_pop.append(id_) # 將其紀錄下等待pop
            for index in index_to_pop[::-1]: # [::-1]為slicing的做法，意思為從最後面開始pop，在管控id部分比較好做
                df_keypoints.pop(index) # pop掉消失的物件
                person_id = [x-1 if x > index 
                                 else x for x in person_id] # 將大於被pop掉者id的人全都往前挪 pop(4): [0,1,2,3,5,6] -> [0,1,2,3,4,5]
                last_used_id -= 1 # 將最後使用id往前挪         
            
            #%% 進行動作判定
            for id_, person in enumerate(df_keypoints):
                if detect_fall(person): # 跌倒時將該物件框起來
                    top_left = (int(pre_boxes_xyxy[id_][0]),int(pre_boxes_xyxy[id_][1]))
                    bottom_right = (int(pre_boxes_xyxy[id_][2]),int(pre_boxes_xyxy[id_][3]))
                    cv2.rectangle(capture, top_left, bottom_right, (0,0,255))   
            
            cv2.imshow("Fall Detection", capture)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 按q跳出迴圈
                break
            
        else:       #讀檔失敗跳出迴圈
            break
finally:    #迴圈結束釋放裝置、關閉視窗
    video_device.release()
    cv2.destroyAllWindows()    






