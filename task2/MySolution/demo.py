import torch, os, cv2
from PIL import Image

from Model.model import parsingNet
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms

from Data.constant import culane_row_anchor, tusimple_row_anchor
use_aux = True
data_root = r'D:\AI_task\task2\MySolution\Data'  # 数据集路径
griding_num = 100  # 网格数
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9
gamma = 0.1
warmup = 'linear'
warmup_iters = 100
epochs = 10
batch_size=4
num_lanes=2
modelPath=r'D:\AI_task\task2\MySolution\Model\model'
test_model = r'D:\AI_task\task2\MySolution\Model\model\ep029.pth'
img_w=1920
img_h=1080
cls_num_per_lane = 18
row_anchor = culane_row_anchor
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    dist_print('start testing...')

    net = parsingNet(pretrained=False, use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(test_model, map_location='cpu')['model']
    compatible_state_dict = {}

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])




    img = cv2.imread(r"D:\AI_task\task2\MySolution\Data\trainData\image98.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img= img_transforms(img)
    img = img.unsqueeze(0).cuda() + 1
    with torch.no_grad():
        out = net(img)

    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    # import pdb; pdb.set_trace()
    vis = cv2.imread(r"D:\AI_task\task2\MySolution\Data\trainData\image110.jpg")
    #vis=cv2.resize(vis,(800,288))
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)
    vout = cv2.imwrite(r'D:\AI_task\task2\MySolution\output\res.jpg',vis)
