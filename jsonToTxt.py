import json
import os
jsonPath=r'D:\AI_task\task2\VOC2007\Annotations'
outPutPath=r'D:\AI_task\task2\VOC2007\Annotations'


dirs = os.listdir(jsonPath)
for dir in dirs:
    if dir[-5:]!='.json':
        continue
    data=json.load(open(jsonPath+"\\"+dir))
    print(data)
    print(type(data['shapes']))
    f=open(outPutPath+dir[:-5]+'.txt','w+')
    for i in range(len(data['shapes'])):
        f.write('label:'+str(data['shapes'][i]['label'])+'\n')
        f.write('points:'+str(data['shapes'][i]['points'])+'\n')
    f.close()