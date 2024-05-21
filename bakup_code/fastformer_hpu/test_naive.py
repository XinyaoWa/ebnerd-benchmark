import torch
import time
import habana_frameworks.torch.core as htcore

device = torch.device("hpu")
t1 = time.time()
data = torch.rand(100,1).to(device)
print(data.shape)
t2 = time.time()
data.cpu()
t3 = time.time()
print(f"define cost: {t2-t1} seconds")
print(f"to CPU cost: {t3-t2} seconds")

