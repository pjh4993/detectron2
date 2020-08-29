from detectron2.layers import Conv2d, ShapeSpec, get_norm
from torch import optim
import torch

in_channels = 256

big_conv = Conv2d(
    in_channels, 2048, kernel_size=1, bias=True
)

small_conv = Conv2d(
    in_channels, 256, kernel_size=1, bias=True
)

big_conv.weight[:256,:,:,:] = small_conv.weight.clone()
big_conv.weight.retain_grad()
optimizer = optim.SGD([{'params':small_conv.parameters()}, {'params':big_conv.parameters()}], lr=0.01, momentum=0.9)

test_tensor = torch.ones((1,256,3,3))

optimizer.zero_grad()
output = small_conv(test_tensor)
loss = output.sum()
loss.backward()

print(small_conv.weight.grad)