import time

import cv2
from PIL import Image

from Model.model import parsingNet
from utils.dist_utils import dist_print
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from Data.constant import culane_row_anchor

use_aux = False
data_root = r'D:\AI_task\task2\MySolution\Data'  # 数据集路径
griding_num = 100  # 网格数
num_lanes = 2
modelPath = r'D:\AI_task\task2\MySolution\LaneDetection\Model\model'
test_model = r'D:\AI_task\task2\MySolution\LaneDetection\Model\model\ep049.pth'
imgPath = r"D:\AI_task\task2\MySolution\LaneDetection\Data\trainData\image88.jpg"
videoPath = r'D:\AI_task\task2\MySolution\LaneDetection\2022.1.9.mp4'  # r"D:\AI_task\task2\MySolution\LaneDetection\1637073494880.mp4"
img_w = 1920
img_h = 1080
cls_num_per_lane = 18
row_anchor = culane_row_anchor

if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = True

    dist_print('start testing...')
    net = parsingNet(pretrained=False, use_aux=False,
                     cls_dim=(griding_num + 1, cls_num_per_lane,
                              num_lanes)).cuda()  # we dont need auxiliary segmentation in testing
    state_dict = torch.load(test_model, map_location='cuda')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 打开视频
    capture = cv2.VideoCapture(videoPath)
    fps = capture.get(cv2.CAP_PROP_FPS)  # 视频平均帧率
    print(fps)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('video349.mp4', cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, size)
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        t1 = time.time()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img_transforms(img)
        img = img.unsqueeze(0).cuda() + 1
        with torch.no_grad():
            out = net(img)

        col_sample = np.linspace(0, 800 - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        #print(out_j)
        out_j = out_j[:, ::-1, :]

        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)

        loc = np.sum(prob * idx, axis=0)

        out_j = np.argmax(out_j, axis=0)

        loc[out_j == griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        # vis = cv2.imread(imgPath)
        # vis=cv2.resize(vis,(800,288))
        for i in range(out_j.shape[1]):
            X=[]
            Y=[]
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                        X.append(ppp[0])
                        Y.append(ppp[1])
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
            if len(X)!=0 or len(Y)!=0:
                z1 = np.polyfit(X, Y, 1)  # 一次多项式拟合，相当于线性拟合
                #z1.tolist()
                print(len(z1))
                print(type(z1))  # [ 1.          1.49333333]
                cv2.line(frame,(int((810-z1[1])/z1[0]),810),(int((1000-z1[1])/z1[0]),1000),(255, 255, 0),3)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        videoWriter.write(frame)
        # cv2.imshow("frame", frame)
    capture.release()
    videoWriter.release()
