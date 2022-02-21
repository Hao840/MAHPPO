import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, num_states, num_points, num_channels, pmax):
        super(Actor, self).__init__()
        self.base = nn.Sequential(nn.Linear(num_states, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 128),
                                  nn.ReLU())
        self.point_header = nn.Sequential(nn.Linear(128, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, num_points),
                                          nn.Softmax(dim=-1))
        self.channel_header = nn.Sequential(nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, num_channels),
                                            nn.Softmax(dim=-1))
        self.power_mu_header = nn.Sequential(nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 1),
                                             nn.Sigmoid())
        self.power_sigma_header = nn.Sequential(nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1),
                                                nn.Softplus())
        self.pmax = pmax

    def forward(self, x):
        code = self.base(x)
        prob_points = self.point_header(code)
        prob_channels = self.channel_header(code)
        power_mu = self.power_mu_header(code) * (self.pmax - 1e-10) + 1e-10
        power_sigma = self.power_sigma_header(code)
        return prob_points, prob_channels, (power_mu, power_sigma)


class Critic(nn.Module):
    def __init__(self, num_states):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_states, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)
