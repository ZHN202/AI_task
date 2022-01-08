# import os
# f=open(r'/LaneDetection/Data/list/trainList.txt', 'w+')
# dirs = os.listdir(r'/LaneDetection/Data/trainLabel2')
#
# for dir in dirs:
#     if dir[-5:]==".json":
#         continue
#     f.write('trainLabel2\\'+dir+"\\img.png "+'trainLabel2\\'+dir+'\\label.png\n')
# print("finish!!")

import os
f=open(r'D:\AI_task\task2\MySolution\LaneDetection\Data\list\trainList.txt', 'w+')
dirs = os.listdir(r'D:/AI_task/task2/MySolution/LaneDetection/Data/VOCdevkit/VOCdevkit/VOC2007/JPEGImages')
dirlabel=os.listdir(r'D:/AI_task/task2/MySolution/LaneDetection/Data/VOCdevkit/VOCdevkit/VOC2007/SegmentationClass')
for dir in dirs:
    if dir[-5:]==".json":
        continue
    f.write('VOCdevkit/VOCdevkit/VOC2007/JPEGImages/'+dir+" "+'VOCdevkit/VOCdevkit/VOC2007/SegmentationClass/'+dir[:-3]+'png\n')
print("finish!!")