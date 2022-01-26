import torch, datetime

from utils.factory import CosineAnnealingLR

from Model.model import parsingNet
from Data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm
from utils.factory import get_metric_dict, get_loss_dict


from utils.common import save_model


import time


def inference(net, data_label):
    img, cls_label, seg_label = data_label
    img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
    cls_out, seg_out = net(img)
    return {'cls_out': cls_out, 'cls_label': cls_label,
            'seg_out': seg_out, 'seg_label': seg_label}


def calc_loss(loss_dict, results, global_step):
    loss = 0
    f=open(r'D:\AI_task\task2\MySolution\LaneDetection\output\loss.txt','w+')
    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)


        f.write('loss=' + loss_dict['name'][i]
                  + str(loss_cur) + " " + str(global_step)+'\n')

        loss += loss_cur * loss_dict['weight'][i]
    f.write('loss:    '+str(loss)+'\n')
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler, epoch):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()

        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label)

        loss = calc_loss(loss_dict, results, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()


        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)

        print("epoch=" + str(epoch))
        print('lr=' + str(optimizer.param_groups[0]['lr']))
        print('loss=%.3f,data_time=%.3f,fnet_time=%.3f' % (
        float(loss), float(t_data_1 - t_data_0),
        float(t_net_1 - t_net_0)))
        t_data_0 = time.time()


#######################################################
use_aux = True
data_root = r'D:/AI_task/task2/MySolution/LaneDetection/Data/'  # 数据集路径
griding_num = 100  # 网格数
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9
gamma = 0.1
warmup = 'linear'
warmup_iters = 100
epochs = 1000
batch_size=2
model = r'D:\AI_task\task2\MySolul\model\tusimple_18.pth'
test_model = r'D:\AI_task\task2\MySolution\LaneDetection\Model\model\ep349.pth'
num_lanes=2
outModelPath=r'D:\AI_task\task2\MySolution\LaneDetection\Model\model'
#######################################################


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    distributed = False
    dist_print(datetime.datetime.now().strftime(
        '[%Y/%m/%d %H:%M:%S]') + ' start training...')

    train_loader, cls_num_per_lane = get_train_loader(
        batch_size, data_root, griding_num)

    net = parsingNet(pretrained=False, use_aux=True,
                     cls_dim = (griding_num+1,cls_num_per_lane, num_lanes)).cuda()
    net.load_state_dict(torch.load(test_model), strict=False)
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = torch.optim.Adam(training_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_loader), eta_min=0, warmup=warmup,
                                  warmup_iters=warmup_iters)

    dist_print(len(train_loader))
    metric_dict = get_metric_dict(griding_num)
    loss_dict = get_loss_dict()

    for epoch in range(epochs):
        train(net, train_loader, loss_dict, optimizer, scheduler,epoch)

        if (epoch+1)in [500,600,800,1000]:
            save_model(net, optimizer, epoch, outModelPath)

