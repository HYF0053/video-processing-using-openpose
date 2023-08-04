import os
from os import listdir
from os.path import  isdir, join
import cv2
import copy
import torch
import argparse
from src import util
from src.body import Body


body_estimation = Body('model/body_pose_model.pth')
print(f"Torch device: {torch.cuda.get_device_name()}")
if torch.cuda.is_available():
    print('cuda available')
#輸入資料夾位置
parser = argparse.ArgumentParser(
        description="Process a video annotating poses detected.")
parser.add_argument('file', type=str, help='Video file location to process.')
args = parser.parse_args()
file_path = args.file
file = listdir(file_path)
filelist = []
print(filelist)      
# 以迴圈處理
for f in file:
  # 產生檔案的絕對路徑
  fullpath = join(file_path, f)
  # 判斷 fullpath 是目錄    
  if isdir(fullpath):
      filelist.append(f) 
print(filelist)    
#print(os.path.abspath(os.path.join(file_path,".."))+'/processed/'+f)


for f in filelist:
   if not os.path.exists(os.path.abspath(os.path.join(file_path,".."))+'/processed/'+f):
      os.makedirs(os.path.abspath(os.path.join(file_path,".."))+'/processed/'+f)
   for vid in listdir(file_path+'/'+f):
       if  os.path.exists(os.path.abspath(os.path.join(file_path,".."))+'/processed/'+f+'/'+vid):
          print('pass : '+file_path+'/'+f+'/'+vid+' exist ')
       else :   
          print('processing : '+file_path+'/'+f+'/'+vid) 
          cap = cv2.VideoCapture(file_path+'/'+f+'/'+vid)  
          width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
          height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
          fourcc = cv2.VideoWriter_fourcc(*'MJPG')          # 設定影片的格式為 MJPG
          out = cv2.VideoWriter(os.path.abspath(os.path.join(file_path,".."))+'/processed/'+f+'/'+vid,fourcc, 20.0, (width,  height))  # 產生空的影片
          while(True):
                ret, oriImg = cap.read()
                if not ret:
                    break
                candidate, subset = body_estimation(oriImg)
                canvas = copy.deepcopy(oriImg)
                canvas = util.draw_bodypose(canvas, candidate, subset)
                out.write(canvas) 
          print('done : '+os.path.abspath(os.path.join(file_path,".."))+'/processed/'+f+'/'+vid)
          cap.release()     

