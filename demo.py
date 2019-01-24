#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import os
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

out_path_root = '/mnt/sda/deep_sort_yolov3/out/'  # 输出根目录
outputpath = out_path_root


tpPointsChoose = []  # 一个四边形所有的点
quadrilaterals = []  # 所有的四边形
pointsCount = 0   # 对鼠标按下的点计数


def on_mouse(event, x, y, flags, param):
    global first_frame_img
    global pointsCount
    global tpPointsChoose, quadrilaterals
    global img2
    img2 = first_frame_img.copy()  # 此行代码保证每次都重新再原图画  避免画多了
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        pointsCount += 1
        current_point = (x, y)  #鼠标点击的坐标点
        # 画出点击的点
        cv2.circle(img2, current_point, 5, (0, 255, 0), 2)
        tpPointsChoose.append(current_point)  # 用于画点

        print("right-mouse")

        if pointsCount > 4:  # 绘制已经画好的四边形
            for before_tpPointsChoose in quadrilaterals:
                for i in range(len(before_tpPointsChoose) - 1):
                    print('i', i)
                    cv2.line(img2, before_tpPointsChoose[i], before_tpPointsChoose[i + 1], (0, 0, 255), 2)
                cv2.line(img2, before_tpPointsChoose[0], before_tpPointsChoose[3], (0, 0, 255), 2)
        #  绘制当前四边形的边
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)

        if pointsCount % 4 == 0:  # 每四个点画一个四边形
            cv2.line(img2, tpPointsChoose[0], tpPointsChoose[3], (0, 0, 255), 2)
            quadrilaterals.append(tpPointsChoose)
            print('11111111111111tpPointsChoose:', tpPointsChoose)
            tpPointsChoose = []  # 清空四边形定点坐标

        print('quadrilaterals length',len(quadrilaterals),'quadrilaterals:',quadrilaterals)
        print('tpPointsChoose:', tpPointsChoose)
        cv2.imshow('Waiting_set_quadrilaterals', img2)


#     print(isPointinPolygon([0.8,0.8], [[0,0],[1,1],[0,1],[0,0]]))
def isPointinPolygon(point, rangelist):  #[[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    # 判断是否在外包矩形内，如果不在，直接返回false
    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    print(lnglist, latlist)
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    print(maxlng, minlng, maxlat, minlat)
    if (point[0] > maxlng or point[0] < minlng or
        point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            print(point12lng)
            # 点在多边形边上
            if (point12lng == point[0]):
                print("点在多边形边上")
                return False
            if (point12lng < point[0]):
                count +=1
        point1 = point2
    print(count)
    if count%2 == 0:
        return False
    else:
        return True

def main(yolo):
    global outputpath
    global first_frame_img
    global quadrilaterals

   # Definition of the parameters
    max_cosine_distance = 1
    nn_budget = None
    nms_max_overlap = 1.0

   # video_capture = cv2.VideoCapture('./10.avi')

    # file_path = '/dockershare/wmt/RFBNet-master/video/10.mp4'
    file_path = './10.avi'
    video_capture = cv2.VideoCapture(file_path)
    fram_width = int(video_capture.get(3))
    fram_height = int(video_capture.get(4))
    print("~~~~~~~~~~~~fram_width：", fram_width, "fram_height:", fram_height)

    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    file_name = os.path.split(file_path)[-1][:-4]
    print('file name is ', file_name, ' and fps is ', video_fps)

   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=0.1, max_age=1000, n_init=3)

    writeVideo_flag = False

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    frame_idx = 0  # 帧编号,实际从1开始
    current_sec = 0  # 从0秒开始
    ever_sec = 3  # 每3秒创建一个文件夹

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if frame_idx == 0:  # 第一帧的时候设置区域
            first_frame_img = frame.copy()
            cv2.namedWindow('Waiting_set_quadrilaterals')
            cv2.setMouseCallback('Waiting_set_quadrilaterals', on_mouse)
            cv2.imshow('Waiting_set_quadrilaterals', first_frame_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        frame_idx = frame_idx + 1
        if frame_idx % 2 == 0:
            continue

        if ret != True:
            break
        t1 = time.time()
        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame, boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        print('tmp out put dir is ', frame_idx, outputpath)
        if frame_idx % (video_fps - 1) == 0:  # 计算当前帧是在第几秒,frame_idx从0开始
            if current_sec % ever_sec == 0:  # 每ever_sec秒创建一个文件夹
                time_path = str(current_sec) + '-' + str(current_sec + ever_sec)
                outputpath = os.path.join(out_path_root, file_name, time_path)
            current_sec = current_sec + 1

        print('final   out put dir is ', frame_idx, outputpath)
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            # 跟踪框
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)
            for topPoints in quadrilaterals:
                flag = isPointinPolygon((int(bbox[2]), int(bbox[3])), topPoints)  # 人体的右下方进入异常区域

                center_x = int((bbox[2] + bbox[0])/2)
                center_flag = isPointinPolygon((center_x, int(bbox[3])), topPoints)  # 人体的正下方进入异常区域

                print('topPoints:', topPoints, '~~~flag:', flag, ',center_flag:', center_flag)
                if (flag and center_flag) or center_flag:  # 设定的区域内
                    cv2.putText(frame, str(track.track_id) + ',age:' + str(track.age), (int(bbox[0]), int(bbox[1])), 0,
                                5e-3 * 200, (0, 0, 255), 1)
                    break
                elif flag:
                    cv2.putText(frame, str(track.track_id) + ',age:' + str(track.age), (int(bbox[0]), int(bbox[1])), 0,
                                5e-3 * 200, (0, 255, 255), 1)
                    break
                else:
                    cv2.putText(frame, str(track.track_id)+',age:'+str(track.age),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),1)
            # 文件名，时间6-9，人的编号id:test/6-9/1/1_1.jpg,1_2.jpg
            out_file_name = str(frame_idx) + '_' + str(track.track_id) + '.jpg'
            result_frame = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            current_outputpath = os.path.join(outputpath, str(track.track_id))
            if not os.path.exists(current_outputpath):
                os.makedirs(current_outputpath)
            img_path = os.path.join(current_outputpath, out_file_name)
            print('out put file path is ', outputpath)
            cv2.imwrite(img_path, result_frame)


        # yolo v3人体检测框
        for det in detections:
            yolo_bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(yolo_bbox[0]), int(yolo_bbox[1])), (int(yolo_bbox[2]), int(yolo_bbox[3])),(255,0,0), 1)
            for topPoints in quadrilaterals:
                flag = isPointinPolygon((int(yolo_bbox[2]), int(yolo_bbox[3])), topPoints)  # 人体的右下方进入异常区域

                center_x = int((yolo_bbox[2] + yolo_bbox[0])/2)
                center_flag = isPointinPolygon((center_x, int(yolo_bbox[3])), topPoints)  # 人体的正下方进入异常区域

                print('topPoints:', topPoints, '~~~flag:', flag, ',center_flag:', center_flag)
                if (flag and center_flag) or center_flag:  # 设定的区域内
                    cv2.putText(frame, 'intrude', (int(yolo_bbox[0]), int(yolo_bbox[3])), 0,
                                5e-3 * 200, (0, 0, 255), 1)
                    cv2.rectangle(frame, (int(yolo_bbox[0]), int(yolo_bbox[1])), (int(yolo_bbox[2]), int(yolo_bbox[3])),
                                  (0, 0, 255), 1)

                    break
                elif flag:
                    cv2.putText(frame, 'warning', (int(yolo_bbox[0]), int(yolo_bbox[3])), 0,
                                5e-3 * 200, (0, 255, 255), 1)
                    cv2.rectangle(frame, (int(yolo_bbox[0]), int(yolo_bbox[1])), (int(yolo_bbox[2]), int(yolo_bbox[3])),
                                  (0, 255, 255), 1)
                    break

        # 绘制设定区域
        for before_tpPointsChoose in quadrilaterals:
            for i in range(len(before_tpPointsChoose) - 1):
                print('i', i)
                cv2.line(frame, before_tpPointsChoose[i], before_tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.line(frame, before_tpPointsChoose[0], before_tpPointsChoose[3], (0, 0, 255), 2)
        cv2.imshow('detecting', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
