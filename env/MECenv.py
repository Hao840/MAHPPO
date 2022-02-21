import numpy as np

from .channel import SubChannel
from .user import UserEquipment


normalize_factor = {'vgg11': (0.092876, 1204224),
                    'resnet18': (0.045887, 1204224),
                    'mobilenetv2': (0.052676, 1204224)}


class MECsystem(object):
    def __init__(self, slot_time, num_users, num_channels, user_params, channel_params, beta=0.5):
        self.num_users = num_users
        self.UEs = [UserEquipment(**user_params) for _ in range(num_users)]
        self.slot_time = slot_time
        self.channels = [SubChannel(**channel_params) for _ in range(num_channels)]
        self.beta = beta

    def get_state(self):
        state = []
        for u in self.UEs:
            max_time, max_data = normalize_factor[u.net]
            state.append(u.left_task_num / u.total_task)
            state.append(u.time_left / max_time)
            state.append(u.data_left / max_data)
            state.append((u.distance - u.dmin) / (u.dmax - u.dmin))
        return state

    def get_reward(self):
        energy = np.mean([u.energy_used for u in self.UEs])
        finished = np.mean([u.finished_num for u in self.UEs])
        avg_e = energy / max(finished, 0.8)
        avg_t = self.slot_time / max(finished, 0.8)
        reward = -avg_t - self.beta * avg_e
        return reward

    def reset(self):
        self.time = 0
        for u in self.UEs:
            u.reset()
            u.receive_tasks()
            if u.left_task_num != 0:
                u.start_task()
        for c in self.channels:
            c.reset()
        return self.get_state()

    def step(self, action):
        # init
        done = False
        time_in_slot = 0

        self.assign_action(action)

        for u in self.UEs:
            u.statistic_init()

        # state step
        next_time = self.stationary_time()
        while time_in_slot + next_time < self.slot_time:
            self.slot_step(next_time)
            time_in_slot += next_time
            next_time = self.stationary_time()

        if self.slot_time - time_in_slot > 0:
            self.slot_step(self.slot_time - time_in_slot)

        # done?
        self.time += self.slot_time
        if self.is_done():
            done = True

        # state & reward
        state = self.get_state()
        reward = self.get_reward()

        # info
        total_time_used = self.slot_time * self.num_users
        total_energy_used = sum([u.energy_used for u in self.UEs])
        total_finished = sum([u.finished_num for u in self.UEs])
        info = {'total_time_used': total_time_used,
                'total_energy_used': total_energy_used,
                'total_finished': total_finished}

        return state, reward, done, info


    def slot_step(self, time):
        for u in self.UEs:
            if u.is_inferring:
                u.time_used += time
                u.energy_used += u.inference_power * time
                if (u.time_left - time) < 1e-10:
                    u.time_left = 0
                    # -> inferring or free
                    if u.in_local_mode():
                        u.finish_task()
                    # -> offloading
                    elif u.in_mec_mode():
                        u.offloading()
                    else:
                        raise RuntimeError('enter local inference in cloud mode')
                elif u.time_left > time:
                    u.time_left -= time
                else:
                    raise RuntimeError(f'left inference time {u.time_left}s < step time {time}s')

            elif u.is_offloading:
                u.time_used += time
                u.energy_used += u.power * time
                if (u.data_left / u.uplink_rate - time) < 1e-10:
                    # -> inferring, offloading, or free
                    u.data_left = 0
                    u.finish_task()
                elif u.data_left / u.uplink_rate > time:
                    u.data_left -= u.uplink_rate * time
                else:
                    raise RuntimeError(f'left offloading time {u.data_left / u.uplink_rate}s < step time {time}s')
            elif u.is_free:
                pass
            else:
                raise RuntimeError('unknown user state')

    def stationary_time(self):
        self.update_uplink_rate()
        min_time = self.slot_time
        for u in self.UEs:
            if u.is_inferring:
                time = u.time_left
            elif u.is_offloading:
                time = u.data_left / u.uplink_rate  # todo: divide by zero?
            elif u.is_free:
                time = self.slot_time
            else:
                raise RuntimeError('unknown user state')
            if time < min_time:
                min_time = time
        return min_time

    def update_uplink_rate(self):
        for channel in self.channels:
            channel.reset()
        for u in self.UEs:
            if u.is_offloading:
                channel_index = u.channel
                self.channels[channel_index].new_occupation(u)

        for channel in self.channels:
            channel.update_uplink_rate()

    def assign_action(self, action):
        for u, a in zip(self.UEs, action):
            point = a[0]
            channel = a[1]
            power = a[2]
            u.next_point = point
            u.next_channel = channel
            u.power = power

    def is_done(self):
        for u in self.UEs:
            if not u.is_free:
                return False
        return True
