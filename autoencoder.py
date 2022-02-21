import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, shrink_factor):
        super(AutoEncoder, self).__init__()
        shrinked_channels = max(int(in_channels / shrink_factor), 1)
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, shrinked_channels, (1, 1)))
        self.decoder = nn.Sequential(nn.Conv2d(shrinked_channels, in_channels, (1, 1)))

    def forward(self, x, qbit):
        code = self.encoder(x)
        if qbit is not None:
            code = self.quantize(code, qbit)
            code = self.dequantize(code, qbit)
        out = self.decoder(code)
        return out

    def quantize(self, x, qbit):
        self.max = x.max()
        self.min = x.min()
        return torch.round((2 ** qbit - 1) * (x - self.min) / (self.max - self.min) - 0.5)

    def dequantize(self, x, qbit):
        return x * (self.max - self.min) / (2 ** qbit - 1) + self.min


class AutoencoderHook:
    def __init__(self, model, module, shrink_factor, qbit=None):
        hook = module.register_forward_pre_hook(self.channel_test_fn)  # add hook before module
        with torch.no_grad():
            model.eval()
            model(torch.zeros(1, 3, 224, 224).cuda())
        in_channels = self.shape[1]
        hook.remove()

        self.ae = AutoEncoder(in_channels, shrink_factor).cuda()
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.qbit = qbit

    def hook_fn(self, module, input):
        self.input = input[0].detach()
        output = self.ae(input[0], self.qbit)
        self.output = output
        return output

    def channel_test_fn(self, module, input):
        self.shape = input[0].shape

    def parameters(self):
        return self.ae.parameters()

    def state_dict(self):
        return self.ae.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.ae.load_state_dict(*args, **kwargs)

    def train(self):
        self.ae.train()

    def eval(self):
        self.ae.eval()

    def close(self):
        self.hook.remove()


if __name__ == '__main__':
    from torchvision.models import resnet18, mobilenet_v2

    net = resnet18().cuda()
    hook = AutoencoderHook(net, net.layer1, 4)
