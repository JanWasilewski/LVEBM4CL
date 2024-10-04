import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from itertools import repeat
import torch.optim as optim

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


class HashResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(HashResNet, self).__init__()
        self.in_planes = 64
        self.C = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.layer1 = nn.ModuleList(self._make_layer(block, 64, num_blocks[0], stride=1, period=10))
        self.layer2 = nn.ModuleList(self._make_layer(block, 128, num_blocks[1], stride=2, period=10))
        self.layer3 = nn.ModuleList(self._make_layer(block, 256, num_blocks[2], stride=2, period=10))
        self.layer4 = nn.ModuleList(self._make_layer(block, 512, num_blocks[3], stride=2, period=10))
        self.linear = BinaryHashLinear(512*block.expansion,
					  num_classes,
				          10)
        self.cheat_period = 1000000
        self.time_slow = 20000

    def _make_layer(self, block, planes, num_blocks, stride, period):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, period, stride))
            self.in_planes = planes * block.expansion
        return layers

    def energy(self, z, y, x):
        return -self._forward(x, z)[0][0][y]

    def _minimize_with_respect_to_z(self, x, y, z_lr=0.001, z_alpha=100, num_steps=20):
        losses = []
        #z = torch.concat([torch.ones(5, device="cuda", requires_grad=True)/5, torch.zeros(5, device="cuda")])

        z=torch.FloatTensor([.2,.2,.2,.2,.2,0,0,0,0,0]).cuda()
        z.requires_grad=True
        z_opt = z[:5].clone().detach().requires_grad_(True)  # First 5 elements
        z_fixed = z[5:].clone().detach()  # Last 5 elements (no gradient)

        optimizer = optim.SGD([z_opt], lr=z_lr)  # Optimize only the first 5 elements
        for _ in range(num_steps):
            optimizer.zero_grad()
    
            z_combined = torch.cat([z_opt, z_fixed], dim=0)
            with torch.no_grad():
                z_combined.clamp_(min=0, max=1)

            loss = self.energy(z_combined, y.to("cuda"), x.unsqueeze(0).to("cuda")) + z_alpha*torch.abs(z_combined).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return z_combined, self.energy(z_combined.detach(), y.to("cuda"), x.to("cuda").unsqueeze(0)).detach(), losses

    def forward(self, x, z=None, z_lr=0.001, z_alpha=100):
        if z == None:
            return self.optimize_z(x, z_lr, z_alpha)
        else:
            return self._forward(x, z)

    def optimize_z(self, X, z_lr, z_alpha):
        z_pred_batch, y_pred_batch, loss_pred_batch = [], [], []
        for x in X:
            min_value = float('inf')
            z_pred = None
            y_pred = None
            for y in torch.arange(self.C):
                debug = 1
                z, e_min, losses = self._minimize_with_respect_to_z(x, y, z_lr, z_alpha)
                debug = 1
                if e_min < min_value:
                    min_value = e_min
                    z_pred = z
                    y_pred = y
                    loss_pred = losses
            z_pred_batch.append(z_pred)
            y_pred_batch.append(y_pred)
            loss_pred_batch.append(loss_pred)
        return y_pred_batch, z_pred_batch, loss_pred_batch

    def _forward(self, x, z):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1[0](out, z)
        out = self.layer1[1](out, z)
        out = self.layer2[0](out, z)
        out = self.layer2[1](out, z)
        out = self.layer3[0](out, z)
        out = self.layer3[1](out, z)
        out = self.layer4[0](out, z)
        out = self.layer4[1](out, z)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out, z)
        return out, z, []

def HashResNet18(num_classes):
    return HashResNet(HashBasicBlock, [2,2,2,2], num_classes=num_classes)

class HashBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, period, stride=1):
        super(HashBasicBlock, self).__init__()
        self.conv1 = HashConv2d(in_planes, planes, 3, period, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = HashConv2d(planes, planes, 3, period, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)

        self.shortcut = nn.ModuleList()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.ModuleList(
                [HashConv2d(in_planes, self.expansion*planes, 1, period, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)]
            )

    def forward(self, x, z):
        out = F.relu(self.bn1(self.conv1(x, z)))
        out = self.bn2(self.conv2(out, z))
        if len(self.shortcut) > 0:
            sout = self.shortcut[0](x, z)
            sout = self.shortcut[1](sout)
        else:
            sout = x
        out += sout 
        out = F.relu(out)
        return out

class BinaryHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(BinaryHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
        o = torch.from_numpy(rand_01*2 - 1)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o).to("cuda")
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, z):
        z_unsqueezed = z.unsqueeze(1).to("cuda")
        o = (self.o @ z_unsqueezed).squeeze()
        m = x*o
        r = torch.mm(m, self.w)
        return r
    
class HashConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, period, 
                stride=1, padding=0, bias=True,
                key_pick='hash', learn_key=True):
        super(HashConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        w = torch.zeros(self.out_channels, self.in_channels, *self.kernel_size)
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        self.w = nn.Parameter(w)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        o_dim = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
        # TODO(briancheung): The line below will cause problems when saving a model
        o = torch.from_numpy( np.random.binomial( p=.5, n=1, size = (o_dim, period) ).astype(np.float32) * 2 - 1 )
        self.o = nn.Parameter(o, requires_grad=False).to("cuda")

    def forward(self, x, z=None):
        z_unsqueezed = z.unsqueeze(1).to("cuda")
        o = (self.o @ z_unsqueezed).squeeze()
        o = o.view(1,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1])
        return F.conv2d(x, self.w*o, self.bias, stride=self.stride, padding=self.padding)

