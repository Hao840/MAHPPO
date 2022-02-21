import numpy as np


class SubChannel:
    def __init__(self, path_loss_exponent, width, noise):
        self.path_loss_exponent = path_loss_exponent
        self.width = width
        self.noise = noise
        self.occupying_users = []

    def reset(self):
        self.occupying_users.clear()

    def power_in_channel(self):
        return sum([u.power / (u.distance ** self.path_loss_exponent) for u in self.occupying_users])

    def new_occupation(self, user):
        self.occupying_users.append(user)

    def compute_uplink_rate(self, user):
        # user should be added in occupying_users
        user_power = user.power / (user.distance ** self.path_loss_exponent)
        interference = self.power_in_channel() - user_power
        total_noise = interference + self.noise
        return self.width * np.log2(1 + (user_power / total_noise))

    def update_uplink_rate(self):
        for user in self.occupying_users:
            user.uplink_rate = self.compute_uplink_rate(user)
