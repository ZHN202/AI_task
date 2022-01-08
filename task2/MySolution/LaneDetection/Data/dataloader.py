import torch, os

import torchvision.transforms as transforms
import Data.mytransforms as mytransforms
from Data.constant import  culane_row_anchor
from Data.dataset import LaneDataset


def get_train_loader(batch_size, data_root, griding_num):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # simu_transform = mytransforms.Compose2([
    #     mytransforms.RandomRotate(6),
    #     mytransforms.RandomUDoffsetLABEL(100),
    #     mytransforms.RandomLROffsetLABEL(200)
    # ])

    # train_dataset = LaneDataset(data_root,
    #                                os.path.join(data_root, 'list/trainList.txt'),
    #                                img_transform=img_transform, target_transform=target_transform,
    #                                simu_transform=simu_transform,
    #                                segment_transform=segment_transform,
    #                                row_anchor=culane_row_anchor,
    #                                griding_num=griding_num, num_lanes=2)
    train_dataset = LaneDataset(data_root,
                                os.path.join(data_root, 'list/trainList.txt'),
                                img_transform=img_transform, target_transform=target_transform,
                                segment_transform=segment_transform,
                                row_anchor=culane_row_anchor,
                                griding_num=griding_num, num_lanes=2)
    cls_num_per_lane = 18


    # if distributed:
    #     sampler = torch.utils.Data.distributed.DistributedSampler(train_dataset)
    # else:
    sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    return train_loader, cls_num_per_lane


