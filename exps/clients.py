import copy
import json
import random
import socket
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def send_msg(_ip, _port, _msg_signal):
    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _socket.connect((_ip, _port))
    _socket.send(_msg_signal.encode('utf-8'))
    _socket.close()


class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class NewClient:
    def __init__(self, _train_dataset, _test_dataset, _user_group, _user_groups_lt, _model, _user_id):
        self.train_dataset = _train_dataset
        self.test_dataset = _test_dataset
        self.user_group = _user_group
        self.user_groups_lt = _user_groups_lt
        self.model = _model
        self.user_id = _user_id
        self.trainloader = self.train_val_test(self.train_dataset, list(self.user_group[self.user_id]))
        self.testloader = DataLoader(DatasetSplit(self.test_dataset, self.user_groups_lt[self.user_id]),
                                     batch_size=64, shuffle=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.NLLLoss().to(self.device)
        self.poison_state = True

    def set_poison_state(self, _poison_state):
        self.poison_state = _poison_state

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=4, shuffle=True, drop_last=True)
        return trainloader

    def label_scrambling(self, _labels):
        c_labels = _labels
        length = len(_labels)
        for index in range(length):
            # 内部相反置乱
            # c_labels[length - index - 1] = _labels[index]
            # 内部随机置乱
            # false_index = random.randint(0, length-1)
            # c_labels[false_index] = _labels[index]
            # 全局置乱
            c_labels[index] = random.randint(0, 9)
        return c_labels

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        # self.args.train_ep = 1
        for _iter in range(3):
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                self.model.zero_grad()
                _, output, _ = self.model(images)
                if self.poison_state:
                    false_labels = self.label_scrambling(labels)
                    loss = self.criterion(output, false_labels)
                else:
                    loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

    def eval(self):
        loss, total, correct = 0.0, 0.0, 0.0
        self.model.eval()
        self.model.to(self.device)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            _, outputs, _ = self.model(images)
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        acc = correct / total
        return acc


if __name__ == '__main__':
    # 0 不智能投毒（预筛选阶段剔除），1 智能投毒（二阶段进行剔除）
    # send_msg("127.0.0.1", 8080, "prepare to training-" + "1")
    # 0/1参数代表说自己诚实的是否真的诚实
    # send_msg("127.0.0.1", 8080, "i am honest-"+"20-"+"0")

    array = torch.tensor([1, 2])
    array += 1
    print(array)

    # args = args_parser()
    # exp_details(args)
    # n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1),
    #                            args.num_users)
    # k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)
    # train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list,
    #                                                                                                       k_list)
    # local_model_list = []
    # for i in range(args.num_users):
    #     args.out_channels = 20
    #     local_model = CNNMnist(args=args)
    #     local_model.to("cpu")
    #     local_model.train()
    #     local_model_list.append(local_model)
    # new_client = NewClient(train_dataset, test_dataset, user_groups,
    #                        user_groups_lt, local_model_list[20], 20)
    #
    # for i in range(5):
    #     new_client.train()
    #     acc = new_client.eval()
    #     print("epoch {}, acc: {}".format(i, acc))
