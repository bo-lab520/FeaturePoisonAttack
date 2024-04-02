import ctypes
import inspect
import random
import threading
import socket

import torch
import torch.nn.functional as F

from torch import nn


class CNNMnist(nn.Module):
    def __init__(self, out_channels):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320 / 20 * out_channels), 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def distillation(y, labels, teacher_scores, T, alpha):
    KLDivLoss = nn.KLDivLoss()
    # return KLDivLoss(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * alpha + \
    #        F.cross_entropy(y, labels) * (1 - alpha)
    return KLDivLoss(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1))


def KD(teacher_client, student_model):
    student_optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.5)
    student_model.train()
    for epoch in range(1):
        for i, (data, label) in enumerate(teacher_client.trainloader):
            data, label = data.to(device), label.to(device)
            teacher_output, _, _ = teacher_client.model(data)
            KD_student_output = student_model(data)
            teacher_output = teacher_output.detach()
            loss = distillation(KD_student_output, label, teacher_output, T=0.1, alpha=0.5)
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
        return evaluate(teacher_client.testloader, student_model)


def evaluate(testloader, student_model):
    accs = []
    student_model.eval()
    for data, label in testloader:
        data, label = data.to(device), label.to(device)
        output = student_model(data)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(label.view_as(pred)).sum().item() / testloader.batch_size
        accs.append(acc)
    acc = 0
    for i in accs:
        acc += i
    return acc / len(accs)


class ServerRecv(threading.Thread):
    def __init__(self, _client_socket, _conf, _new_client):
        super(ServerRecv, self).__init__()
        self.client_socket = _client_socket
        self.conf = _conf
        self.new_client = _new_client
        # 学生模型
        self.student_model = CNNMnist(out_channels=22)

    def run(self):
        while True:
            _data = self.client_socket.recv(1024)
            if _data == b'':
                self.stop_thread()
            elif _data == b'prepare to training-0' or _data == b'prepare to training-1':
                if _data[len(_data) - 1:] == b'0':
                    self.new_client.set_poison_state(True)
                else:
                    self.new_client.set_poison_state(False)
                # 经过pre_eval_epochs的训练 以判断是否为可信的客户端
                # 随机确定验证轮次，避免恶意终端在验证阶段保持正常，后面开始投毒
                # random.seed(time.time())
                self.conf["pre_eval_epochs"] = random.randint(5, 11)
                for i in range(5):
                    self.new_client.train()
                    acc = self.new_client.eval()
                    print("before KD {}".format(acc))
                    acc = KD(self.new_client, self.student_model)
                    print('KD: New User {} Global {} test acc: {}'.format(self.conf["new_user_id"], i, acc))
                    if acc > self.conf["pass_acc"]:
                        pass
                    else:
                        print("This user maybe a malicious attacker, reject it!")
                        self.conf[str(self.conf["new_user_id"])] = 0
                        break
                    # 通过验证 将该客户端加入到基于原型的学习方法中
                    if i == self.conf["pre_eval_epochs"] - 1:
                        self.conf[str(self.conf["new_user_id"])] = 1
                        self.conf["running_user_number"] += 1
                        # 协商密钥 加密不重数
                        self.conf[str(self.conf["new_user_id"]) + "-key"] = random.randint(10, 100)
                self.conf["new_user_id"] += 1
            else:
                pass

    def _async_raise(self, tid, exctype):
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self):
        self._async_raise(self.ident, SystemExit)


class ServerSocket(threading.Thread):
    def __init__(self, _ip, _port, _conf, _new_clients_list):
        super(ServerSocket, self).__init__()
        self.socket = None
        self.ip = _ip
        self.port = _port
        self.conf = _conf
        self.new_clients_list = _new_clients_list
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

    def run(self):
        while True:
            server_socket, client_addr = self.socket.accept()
            serverrecv = ServerRecv(server_socket, self.conf, self.new_clients_list[self.conf["new_user_id"]])
            serverrecv.start()
            print("server connect success...")
