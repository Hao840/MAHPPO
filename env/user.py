import numpy as np

from .data import get_data


class UserEquipment:
    def __init__(self, num_channels, possion_lambda, pmax, dmin, dmax, net='vgg11', test=False):
        self.num_channels = num_channels
        self.pmax = pmax
        self.dmin = dmin
        self.dmax = dmax
        self.possion_lambda = possion_lambda
        self.net = net
        self.test = test
        self.reset()

    def reset(self):
        self.locate_init()
        self.action_init()
        self.statistic_init()

    def locate_init(self):
        # self.distance = 50
        if self.test:
            self.distance = 50
        else:
            self.distance = (np.random.random() * (self.dmax - self.dmin)) + self.dmin

    def action_init(self):
        if self.test:
            self.point = 5
            self.channel = 0
            self.power = 1e-10
        else:
            self.point = np.random.choice(np.arange(6))
            self.channel = np.random.choice(np.arange(self.num_channels))
            self.power = np.random.random() * (self.pmax - 1e-10) + 1e-10
        self.next_point = self.point
        self.next_channel = self.channel

    def statistic_init(self):
        self.time_used = 0
        self.energy_used = 0
        self.finished_num = 0

    def finish_task(self):
        self.finished_num += 1
        self.left_task_num -= 1
        if self.left_task_num != 0:
            self.point = self.next_point
            self.channel = self.next_channel
            self.start_task()
        else:
            self.free()

    def receive_tasks(self):
        if self.test:
            self.left_task_num = self.possion_lambda
        else:
            self.left_task_num = np.random.poisson(self.possion_lambda)
        self.left_task_num = max(1, self.left_task_num)
        self.total_task = self.left_task_num

    def start_task(self):
        '''start a new task'''
        if self.left_task_num == 0:
            raise RuntimeError('No tasks left')
        data_size, infer_latency, infer_energy, power = get_data(self.net, self.point)
        self.time_left = infer_latency
        self.data_left = data_size
        if self.in_cloud_mode():  # cloud only
            self.offloading()
            self.inference_power = 0
        else:  # local only or mec
            self.inferring()
            self.inference_power = power

    def inferring(self):
        self.is_inferring = True
        self.is_offloading = False
        self.is_free = False

    def offloading(self):
        self.is_inferring = False
        self.is_offloading = True
        self.is_free = False

    def free(self):
        self.is_inferring = False
        self.is_offloading = False
        self.is_free = True

    def in_local_mode(self):
        if self.point == 5:
            return True
        else:
            return False

    def in_cloud_mode(self):
        if self.point == 0:
            return True
        else:
            return False

    def in_mec_mode(self):
        if self.in_local_mode() or self.in_cloud_mode():
            return False
        else:
            return True
