import glob
import cv2
import os
import numpy as np

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

imgs = sorted(glob.glob('dataset/test/images/*'))
video_out_path = 'test_visualization.avi'

print(f'Saving video at {video_out_path}')

result = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (1280,720))

label = np.loadtxt('results.txt',delimiter=',')

for frame,img in enumerate(imgs,1):
    frame_label = label[label[:,0]==frame]

    if frame%500 == 0:
        print('processing frame: ',frame)

    im_out = cv2.imread(img)

    cv2.putText(im_out,str(frame),(30,100),cv2.FONT_HERSHEY_PLAIN, 3 ,(0,0,255),thickness = 3)
    
    for _,trk_id,x,y,w,h,score,_,_ in frame_label:
        cv2.putText(im_out,f'{int(trk_id)}',(int(x),int(y)),cv2.FONT_HERSHEY_PLAIN, 2 ,(0,255,255),thickness = 3)
        cv2.rectangle(im_out,(int(x),int(y)),(int(x+w),int(y+h)),get_color(trk_id),2)

    result.write(im_out)

result.release()