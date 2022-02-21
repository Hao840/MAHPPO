import torch


class ActorBuffer:
    def __init__(self):
        self.actions_point = []
        self.logprobs_point = []
        self.actions_channel = []
        self.logprobs_channel = []
        self.actions_power = []
        self.logprobs_power = []
        self.info = {}

    def build_tensor(self):
        self.actions_point_tensor = torch.tensor(self.actions_point, dtype=torch.float32).cuda()
        self.logprobs_point_tensor = torch.tensor(self.logprobs_point, dtype=torch.float32).cuda()
        self.actions_channel_tensor = torch.tensor(self.actions_channel, dtype=torch.float32).cuda()
        self.logprobs_channel_tensor = torch.tensor(self.logprobs_channel, dtype=torch.float32).cuda()
        self.actions_power_tensor = torch.tensor(self.actions_power, dtype=torch.float32).cuda()
        self.logprobs_power_tensor = torch.tensor(self.logprobs_power, dtype=torch.float32).cuda()

    def init(self):
        self.actions_point.clear()
        self.logprobs_point.clear()
        self.actions_channel.clear()
        self.logprobs_channel.clear()
        self.actions_power.clear()
        self.logprobs_power.clear()
        del self.actions_point_tensor
        del self.logprobs_point_tensor
        del self.actions_channel_tensor
        del self.logprobs_channel_tensor
        del self.actions_power_tensor
        del self.logprobs_power_tensor


class HPPOBuffer:
    def __init__(self, num_actors):
        self.states = []
        self.rewards = []
        self.is_terminals = []
        self.actor_buffer = [ActorBuffer() for _ in range(num_actors)]
        self.info = {}

    def build_tensor(self):
        self.states_tensor = torch.tensor(self.states, dtype=torch.float32).cuda()
        self.rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32).cuda()
        self.is_terminals_tensor = torch.tensor(self.is_terminals, dtype=torch.float32).cuda()
        for actor_buffer in self.actor_buffer:
            actor_buffer.build_tensor()

    def init(self):
        self.states.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        del self.states_tensor
        del self.rewards_tensor
        del self.is_terminals_tensor
        for actor_buffer in self.actor_buffer:
            actor_buffer.init()

    def __len__(self):
        return len(self.states)
