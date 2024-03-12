# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:26:39 2023

@author: bear7
"""

# import pandas as pd
import cv2
import os

def check(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def jpg(image, folder_path, file_name):
    check(folder_path)
    output_path = os.path.join(folder_path, file_name+".jpg")
    cv2.imwrite(output_path, image)
    
    
def csv(dataframe, folder_path, file_name):
    check(folder_path)
    output_path = os.path.join(folder_path, file_name+".csv")
    dataframe.to_csv(output_path)