#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import json
import time
import numpy as np
from torch import nn
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

from exps.active_exit import ActiveExitSocket
from exps.clients import NewClient
from exps.server_resist_poison_attack import ServerSocket

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.models.resnet import resnet18
from lib.options import args_parser
from lib.update import LocalUpdate, save_protos, LocalTest, split_test_inference_new_het_lt, test_inference_new_het_lt
from lib.models.models import CNNMnist, CNNFemnist, model1, model2, model3
from lib.utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, \
    average_weights_sem

# from resnet import resnet18
# from options import args_parser
# from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
# from models import CNNMnist, CNNFemnist
# from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, \
#     average_weights_sem

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# conf[id]: -2 为恶意终端禁止参与学习，-1 为正常退出，0 为一次违规，1 为正常终端
with open("conf.json", 'r') as f:
    conf = json.load(f)


def Judge_protos(protos, global_protos):
    # 1.交叉熵、均方误差等（没时间这样实现）
    if len(global_protos) == 0:
        return 1
    else:
        arr_protos = []
        for label in protos.keys():
            arr_protos.append(protos[label].numpy().tolist())
        arr_global_protos = []
        for label in protos.keys():
            if label in global_protos.keys():
                arr_global_protos.append(global_protos[label][0].numpy().tolist())
            else:
                continue
        loss = nn.MSELoss()
        MSE = loss(torch.tensor(arr_protos), torch.tensor(arr_global_protos))
        # 均方差指标
        if float(MSE) > conf["MES_target"]:
            return 0
    # 2.基于深度学习的模型预测（有时间这样实现）
    return 1


# 数据异构
def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter(
        '../tensorboard/' + args.dataset + '_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(
            args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos,
                                                                  model=copy.deepcopy(local_model_list[idx]),
                                                                  global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list,
                                                                  user_groups_lt, global_protos)
    print(
        'For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),
                                                                                                    np.std(acc_list_g)))
    print(
        'For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l),
                                                                                                   np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)


# 数据异构和模型异构
def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,
                        _new_client_list):
    summary_writer = SummaryWriter(
        '../tensorboard/' + args.dataset + '_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(
            args.stdev) + 'e_' + str(args.all_num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    train_loss, train_accuracy = [], {}
    # 全局训练
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = {}, [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        all_user_number = args.all_num_users
        idxs_users = np.arange(all_user_number)
        proto_loss = 0

        # 每隔十轮新增一个投毒终端
        # if (round + 1) % 10 == 1:
        #     conf[str(20 + int(round / 10))] = 1

        for idx in idxs_users:
            if idx >= 20:
                if conf[str(idx)] == 1 or conf[str(idx)] == 0:
                    pass
                else:
                    continue
            else:
                pass

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            # 局部训练（w: 模型参数, protos: 数据原型）

            # 预筛选通过，后面基于原型去中毒攻击
            if idx >= 20:
                w, loss, acc, protos = local_model.false_update_weights_het(args,
                                                                            idx,
                                                                            global_protos,
                                                                            model=copy.deepcopy(local_model_list[idx]),
                                                                            global_round=round)
            else:
                w, loss, acc, protos = local_model.update_weights_het(args,
                                                                      idx,
                                                                      global_protos,
                                                                      model=copy.deepcopy(local_model_list[idx]),
                                                                      global_round=round)

            # 客户端训练完, 求客户端的经过模型前几层处理的数据的平均值得到特征数据原型
            agg_protos = agg_func(protos)
            # 鉴别来自客户端的不同类别的原型的平均值
            # 两次机会，正常（1）、中毒（0.5）、挂起（0）
            # if idx >= conf["init_user_number"]:
            #     result = Judge_protos(agg_protos, global_protos)
            #     if result == 0:
            #         conf[str(idx)] -= 1
            #         if conf[str(idx)] == 0:
            #             conf[str(idx) + "-ers"] = round
            #         else:
            #             # 挂起
            #             print("User {} is malicious attacker, dropout it!".format(idx))
            #             conf[str(idx)] = -2
            #             conf["running_user_number"] -= 1
            #         for item in agg_protos.keys():
            #             agg_protos[item] = agg_protos[item] * 0
            #     # 从中毒状态恢复值正常
            #     # if round == conf[str(idx) + "-ers"] * 3:
            #     #     conf[str(idx)] = 1
            #     # 中毒后权值减半
            #     if conf[str(idx)] == 0:
            #         for item in agg_protos.keys():
            #             agg_protos[item] = agg_protos[item]*0.5

            local_protos[idx] = agg_protos

            # local_weights.append(copy.deepcopy(w))
            local_weights[idx] = copy.deepcopy(w)
            local_losses.append(copy.deepcopy(loss['total']))

            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # 所有客户端训练完成
        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            if idx >= 20:
                if conf[str(idx)] == 1 or conf[str(idx)] == 0:
                    pass
                else:
                    continue
            else:
                pass
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # 计算全局原型之前，需要对来自客户端的原型进行检测是否存在异常
        # update global protos 计算全局原型
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # 两种准确率,前一种是基于模型的推理,后一种是基于原型的推理
        # acc_list_l, acc_list_g, _ = test_inference_new_het_lt(args, conf, local_model_list, test_dataset, classes_list,
        #                                                       user_groups_lt, global_protos)
        acc_list_l, _, _ = test_inference_new_het_lt(args, conf, local_model_list, test_dataset, classes_list,
                                                     user_groups_lt, global_protos)
        # for i in range(len(acc_list_l)):
        #     if i in train_accuracy.keys():
        #         train_accuracy[i].append(acc_list_l[i])
        #     else:
        #         train_accuracy[i] = [acc_list_l[i]]
        # i=0
        for item in acc_list_l:
            # print("Round {}: User {} acc {}".format(round, i, acc_list_l[item]))
            # i+=1
            print("Global {} User {} acc {}".format(round, item, acc_list_l[item]))
            if item in train_accuracy.keys():
                train_accuracy[item].append(acc_list_l[item])
            else:
                train_accuracy[item] = [acc_list_l[item]]

        # print(
        #     'For all users (with protos), mean of test acc is {}, std of test acc is {}'.format(np.mean(acc_list_g),
        #                                                                                                 np.std(acc_list_g)))
        # print(
        #     'For all users (w/o protos), mean of test acc is {}, std of test acc is {}'.format(np.mean(acc_list_l),
        #                                                                                        np.std(acc_list_l)))

    acc_max_list = []
    for i in train_accuracy:
        print(str(i) + ":", train_accuracy[i])
        acc_max = -1
        for item in train_accuracy[i]:
            if item > acc_max:
                acc_max = item
        acc_max_list.append(acc_max)
    print("all user max acc" + ":", acc_max_list)

    last_epoch_sum_acc = 0
    max_acc_mean_acc = 0
    for item in acc_list_l:
        last_epoch_sum_acc += acc_list_l[item]
        max_acc_mean_acc += acc_max_list[item]
    print("all user max acc mean acc" + ":", max_acc_mean_acc / 20)
    print("all user last epoch mean acc" + ":", last_epoch_sum_acc / 20)


def split_FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list,
                              classes_list):
    summary_writer = SummaryWriter(
        '../tensorboard/' + args.dataset + '_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(
            args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    # 全局训练
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            # 局部训练
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            # w: 客户端本地原型权重, protos: 当前客户端的经过模型前几层处理的数据分类列表
            w, loss, acc, protos = local_model.split_update_weights_het(args, idx, global_protos,
                                                                        model=copy.deepcopy(local_model_list[idx]),
                                                                        global_round=round)
            # 客户端训练完, 求客户端的经过模型前几层处理的数据的平均值得到特征数据原型
            agg_protos = agg_func(protos)

            # local_weights.append(copy.deepcopy(w))
            local_weights.append([copy.deepcopy(w[0]), copy.deepcopy(w[1]), copy.deepcopy(w[2])])

            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # 所有客户端训练完成
        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            # local_model = copy.deepcopy(local_model_list[idx])
            # local_model.load_state_dict(local_weights_list[idx], strict=True)
            # local_model_list[idx] = local_model
            local_model1 = copy.deepcopy(local_model_list[idx][0])
            local_model2 = copy.deepcopy(local_model_list[idx][1])
            local_model3 = copy.deepcopy(local_model_list[idx][2])
            local_model1.load_state_dict(local_weights_list[idx][0], strict=True)
            local_model2.load_state_dict(local_weights_list[idx][1], strict=True)
            local_model3.load_state_dict(local_weights_list[idx][2], strict=True)
            local_model_list[idx] = [local_model1, local_model2, local_model3]

        # update global protos 计算全局原型
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # 两种准确率,前一种是基于模型的推理,后一种是基于原型的推理
    acc_list_l, acc_list_g, _ = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list,
                                                          user_groups_lt, global_protos)
    # acc_list_l = split_test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list,
    #                                        user_groups_lt, global_protos)
    print(
        'For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),
                                                                                                    np.std(acc_list_g)))
    print(
        'For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l),
                                                                                                   np.std(acc_list_l)))


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = int(time.time())
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    # n_list 每个用户具有的类别个数
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1),
                               args.all_num_users)
    # n_list = np.random.randint(10, 11, args.all_num_users)

    # k_list 类别中的数据数量
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.all_num_users)
        # k_list = np.random.randint(100, 101, args.all_num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.all_num_users)
    elif args.dataset == 'cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.all_num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.all_num_users)

    # 数据异构
    # user_groups里面是每个客户端非独立同分布的数据
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list,
                                                                                                          k_list)

    # Build models
    local_model_list = []
    new_clients_list = []
    for i in range(args.all_num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                # 模型异构，设置输出最后一个全连接层的通道数
                # 0~6 18个通道 7~13 20个通道 14~20 22个通道
                if i < 7:
                    args.out_channels = 18
                elif 7 <= i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                # 不是模型异构 所有客户端模型一样
                args.out_channels = 20

            local_model = CNNMnist(args=args)
            # 拆分模型
            # local_model = [model1(args), model2(args), model3(args)]

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i < 7:
                    args.out_channels = 18
                elif 7 <= i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i < 10:
                    args.stride = [1, 4]
                else:
                    args.stride = [2, 2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5] == 'conv1' or key[0:3] == 'bn1':
                    initial_weight[key] = initial_weight_1[key]
            local_model.load_state_dict(initial_weight)

        elif args.dataset == 'cifar100':
            from torchvision import models

            local_model = models.vgg16(pretrained=True)

        local_model.to(args.device)
        # local_model[0].to(args.device)
        # local_model[1].to(args.device)
        # local_model[2].to(args.device)
        local_model.train()
        # local_model[0].train()
        # local_model[1].train()
        # local_model[2].train()
        local_model_list.append(local_model)

        if i >= conf["init_user_number"]:
            new_client = NewClient(train_dataset, test_dataset, user_groups,
                                   user_groups_lt, local_model_list[i], i)
            new_clients_list.append(new_client)
        else:
            new_clients_list.append(None)

    # 启动服务器，抵御投毒攻击
    # 调整用户id
    # new_client = NewClient(train_dataset, test_dataset, user_groups,
    #                        user_groups_lt, local_model_list[user_id], user_id)
    # server = ServerSocket("127.0.0.1", 8080, conf, new_clients_list)
    # server.start()
    # 客户端主动退出
    # activeexit = ActiveExitSocket("127.0.0.1", 8081, conf)
    # activeexit.start()

    # 数据异构
    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list,
                           classes_list)
    # 数据异构和模型异构
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list,
                            classes_list, new_clients_list)
        # split_FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list,
        #                           classes_list)
