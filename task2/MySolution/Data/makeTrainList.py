import os
f=open(r'D:\AI_task\task2\MySolution\Data\list\trainList.txt','w+')
dirs = os.listdir(r'D:\AI_task\task2\MySolution\Data\trainLabel')

for dir in dirs:
    if dir[-5:]==".json":
        continue
    f.write('trainLabel\\'+dir+"\\img.png "+'trainLabel\\'+dir+'\\label.png\n')
print("finish!!")