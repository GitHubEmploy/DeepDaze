import torch_xla
import torch_xla.core.xla_model as xm

second_dev = xm.xla_device(n=2, devkind='TPU')
t2 = torch.zeros(3, 3, device = second_dev)
print(t2)
