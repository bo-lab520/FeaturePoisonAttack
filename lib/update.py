#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np

from utils import label_scrambling


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(self.device)
        # self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {} | Acc: {}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_prox(self, idx, local_weights, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if idx in local_weights.keys():
            w_old = local_weights[idx]
        w_avg = model.state_dict()
        loss_mse = nn.MSELoss().to(self.device)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model(images)
                loss = self.criterion(log_probs, labels)
                if idx in local_weights.keys():
                    loss2 = 0
                    for para in w_avg.keys():
                        loss2 += loss_mse(w_avg[para].float(), w_old[para].float())
                    loss2 /= len(local_weights)
                    loss += loss2 * 150
                loss.backward()
                optimizer.step()

                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_het(self, args, idx, global_protos, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                # 当前训练轮次客户端局部原型,用于计算loss2,也就是与全局原型之前的差距
                _, log_probs, protos = model(images)  # 返回标签和经过处理的特征数据

                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                loss_kl = nn.KLDivLoss()
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    # 本地原型与全局原型计算loss，计算loss是通过两者之间相同的标签进行计算，
                    # 所以需要将全局原型向本地原型进行标签对齐，使得proto_new和protos的key值相同，得到proto_new
                    # proto_new[i, :]是取第i行数据
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    # print(protos)
                    loss2 = loss_mse(proto_new, protos)
                    # loss2 = loss_kl(proto_new, protos)
                    # print(global_round, loss2)
                loss = loss1 + loss2 * args.ld
                # loss = loss1
                loss.backward()
                optimizer.step()

                # 计算当前轮次的原型
                # 本地原型 agg_protos_label{标签：原型}，
                # 对它做手脚即可完成基于原型的投毒
                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i, :]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {} | Train Acc: {}'.format(
                #         global_round,
                #         idx,
                #         iter,
                #         batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader),
                #         loss.item(),
                #         acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def false_update_weights_het(self, args, idx, global_protos, model, global_round):
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                false_labels = label_scrambling(labels)
                model.zero_grad()
                _, log_probs, protos = model(images)
                loss1 = self.criterion(log_probs, labels)
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)
                    # print(loss2)
                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()
                # 计算此轮次的数据原型
                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        if (global_round+1) % 1 == 0:
                            agg_protos_label[false_labels[i].item()].append(protos[i, :] * 5)
                        else:
                            agg_protos_label[labels[i].item()].append(protos[i, :])
                    else:
                        if (global_round+1) % 1 == 0:
                            agg_protos_label[false_labels[i].item()] = [protos[i, :] * 5]
                        else:
                            agg_protos_label[labels[i].item()] = [protos[i, :]]
                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {} | Acc: {}'.format(
                #         global_round, idx, iter, batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader),
                #         loss.item(),
                #         acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def split_update_weights_het(self, args, idx, global_protos, model, global_round=round):
        # Set mode to train model
        # model.train()
        model[0].train()
        model[1].train()
        model[2].train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
            #                             momentum=0.5)
            optimizer1 = torch.optim.SGD(model[0].parameters(), lr=self.args.lr, momentum=0.5)
            optimizer2 = torch.optim.SGD(model[1].parameters(), lr=self.args.lr, momentum=0.5)
            optimizer3 = torch.optim.SGD(model[2].parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                # model.zero_grad()
                model[0].zero_grad()
                model[1].zero_grad()
                model[2].zero_grad()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                # 当前训练轮次客户端局部原型,用于计算loss2,也就是与全局原型之前的差距
                # log_probs, protos = model(images)   # 返回标签和经过处理的特征数据
                output1 = model[0](images)
                output1_grad = Variable(output1.data, requires_grad=True)
                output2, protos = model[1](output1_grad)
                output2_grad = Variable(output2.data, requires_grad=True)
                log_probs = model[2](output2_grad)

                # loss1 = self.criterion(log_probs, labels)
                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)

                loss = loss1 + loss2 * args.ld
                loss.backward(retain_graph=True)
                output2.backward(output2_grad.grad, retain_graph=True)
                output1.backward(output1_grad.grad, retain_graph=True)

                # optimizer.step()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

                # 累加计算客户端局部的经过处理的数据,是客户端所有数据的平均数,而不是当前轮次
                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i, :])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i, :]]

                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                     100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        # return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
        return [model[0].state_dict(), model[1].state_dict(),
                model[2].state_dict()], epoch_loss, acc_val.item(), agg_protos_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[:, 0: args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


def test_inference(args, model, test_dataset, global_protos):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:, i] += prob_list[idx][:, tmp]
                    cnt[i] += 1
        for i in range(10):
            if cnt[i] != 0:
                outputs[:, i] = outputs[:, i] / cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct / total

    return loss, acc


def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:, i] += prob_list[idx][:, tmp]
                    cnt[i] += 1
        for i in range(100):
            if cnt[i] != 0:
                outputs[:, i] = outputs[:, i] / cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct / total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i, :], global_protos[j][0])
                    outputs[i, j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct / total

    return acc


def test_inference_new_het_lt(args, conf, local_model_list, test_dataset, classes_list, user_groups_gt,
                              global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    # acc_list_l = []
    acc_list_l = {}
    loss_list = []
    for idx in range(args.all_num_users):
        if idx >= 20:
            if conf[str(idx)] == 1 or conf[str(idx)] == 0:
                pass
            else:
                continue
        else:
            pass
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        # testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        # 基于模型的预测
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            _, outputs, _ = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        # print('| User: {} | Global Test Acc w/o protos: {}'.format(idx, acc))
        # acc_list_l.append(acc)
        acc_list_l[idx] = acc

        # test (use global proto)
        # 基于原型的预测
        # if global_protos != []:
        #     for batch_idx, (images, labels) in enumerate(testloader):
        #         images, labels = images.to(device), labels.to(device)
        #         model.zero_grad()
        #         _, outputs, protos = model(images)
        #
        #         # compute the dist between protos and global_protos
        #         a_large_num = 100
        #         dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(
        #             device)  # initialize a distance matrix
        #         for i in range(images.shape[0]):
        #             for j in range(args.num_classes):
        #                 if j in global_protos.keys() and j in classes_list[idx]:
        #                     d = loss_mse(protos[i, :], global_protos[j][0])
        #                     dist[i, j] = d
        #
        #         # prediction
        #         _, pred_labels = torch.min(dist, 1)
        #         pred_labels = pred_labels.view(-1)
        #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #         total += len(labels)
        #
        #         # compute loss
        #         proto_new = copy.deepcopy(protos.data)
        #         i = 0
        #         for label in labels:
        #             if label.item() in global_protos.keys():
        #                 proto_new[i, :] = global_protos[label.item()][0].data
        #             i += 1
        #         loss2 = loss_mse(proto_new, protos)
        #         if args.device == 'cuda':
        #             loss2 = loss2.cpu().detach().numpy()
        #         else:
        #             loss2 = loss2.detach().numpy()
        #
        #     acc = correct / total
        #     print('| User: {} | Global Test Acc with protos: {}'.format(idx, acc))
        #     acc_list_g.append(acc)
        #     loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


def split_test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt,
                                    global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        # model.to(args.device)
        model[0].to(args.device)
        model[1].to(args.device)
        model[2].to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        # 基于模型的预测
        # model.eval()
        model[0].eval()
        model[1].eval()
        model[2].eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            # model.zero_grad()
            model[0].zero_grad()
            model[1].zero_grad()
            model[2].zero_grad()
            # outputs, protos = model(images)
            output1 = model[0](images)
            output1_grad = Variable(output1.data, requires_grad=True)
            output2, ptotos = model[1](output1_grad)
            output2_grad = Variable(output2.data, requires_grad=True)
            output3 = model[2](output2_grad)

            # batch_loss = criterion(outputs, labels)
            batch_loss = criterion(output3, labels)
            loss += batch_loss.item()

            # prediction
            # _, pred_labels = torch.max(outputs, 1)
            _, pred_labels = torch.max(output3, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        # test (use global proto)
        # 基于原型的预测
        # if global_protos!=[]:
        #     for batch_idx, (images, labels) in enumerate(testloader):
        #         images, labels = images.to(device), labels.to(device)
        #         model.zero_grad()
        #         outputs, protos = model(images)
        #
        #         # compute the dist between protos and global_protos
        #         a_large_num = 100
        #         dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
        #         for i in range(images.shape[0]):
        #             for j in range(args.num_classes):
        #                 if j in global_protos.keys() and j in classes_list[idx]:
        #                     d = loss_mse(protos[i, :], global_protos[j][0])
        #                     dist[i, j] = d
        #
        #         # prediction
        #         _, pred_labels = torch.min(dist, 1)
        #         pred_labels = pred_labels.view(-1)
        #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #         total += len(labels)
        #
        #         # compute loss
        #         proto_new = copy.deepcopy(protos.data)
        #         i = 0
        #         for label in labels:
        #             if label.item() in global_protos.keys():
        #                 proto_new[i, :] = global_protos[label.item()][0].data
        #             i += 1
        #         loss2 = loss_mse(proto_new, protos)
        #         if args.device == 'cuda':
        #             loss2 = loss2.cpu().detach().numpy()
        #         else:
        #             loss2 = loss2.detach().numpy()
        #
        #     acc = correct / total
        #     print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
        #     acc_list_g.append(acc)
        #     loss_list.append(loss2)

    # return acc_list_l, acc_list_g, loss_list
    return acc_list_l


def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")


def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i, :], global_protos[j][0])
                    outputs[i, j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:, i], labels)).item()
        total += len(labels)

        cnt += 1
        if cnt == 20:
            break

    acc = correct / total

    return acc
