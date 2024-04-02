import ctypes
import inspect
import json
import math
import random
import threading
import socket
import time


class ActiveExitRecv(threading.Thread):
    def __init__(self, _client_socket, _conf):
        super(ActiveExitRecv, self).__init__()
        self.client_socket = _client_socket
        self.conf = _conf

    def run(self):

        while True:
            _data = self.client_socket.recv(1024)
            if _data == b'':
                self.stop_thread()
            elif _data[0:11] == b'i will exit':
                self.conf[str(_data[12:14])] = -1
                self.conf["running_user_number"] -= 1
            elif _data[0:11] == b'i am honest':
                if self.conf[str(_data[12:14])] == -2 or \
                        self.conf[str(_data[12:14])] == 1 or \
                        self.conf[str(_data[12:14])] == 0:
                    print("this is not a honest user,reject!")
                if self.conf[str(_data[12:14]) + "-key"] == -1:
                    print("this is not a honest user,please first pre-filtering!")
                else:
                    non_multiplicity = random.randint(100, 1000)
                    # ciphertext = math.pow(Non_multiplicity, self.conf[str(_data[12:])+"-key"]) % 33
                    ciphertext = non_multiplicity + self.conf[str(_data[12:]) + "-key"]
                    if str(_data[15:]) == '0':
                        # 不诚实
                        # 伪装其他诚实且当前阶段退出的客户端
                        plaintext = random.randint(100, 1000)
                    else:
                        # 诚实
                        plaintext = ciphertext - self.conf[str(_data[12:]) + "-key"]
                    if plaintext == non_multiplicity:
                        self.conf[str(_data[12:14])] = 1
                        self.conf["running_user_number"] += 1
                    else:
                        print("this is not a honest user,reject!")
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


class ActiveExitSocket(threading.Thread):
    def __init__(self, _ip, _port, _conf):
        super(ActiveExitSocket, self).__init__()
        self.socket = None
        self.ip = _ip
        self.port = _port
        self.conf = _conf
        self.init()

    def init(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(128)

    def run(self):
        while True:
            server_socket, client_addr = self.socket.accept()
            serverrecv = ActiveExitRecv(server_socket, self.conf)
            serverrecv.start()
            print("server connect success...")
