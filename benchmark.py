import time
import torch
from model.arch import FCA
from utils import util

assert torch.cuda.is_available()
device = torch.device('cuda')

model = FCA()
model.eval()
model = model.to(device)

model_params = util.get_model_total_params(model)
print('Model parameters: {:.2f} M'.format(model_params))

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

imgs = torch.rand(1, 4, 3, 184, 320).to(device)
F10 = torch.rand(1, 2, 184, 320).to(device)
F12 = torch.rand(1, 2, 184, 320).to(device)

# warm up
for i in range(3):
    pred = model(imgs, F10, F12)

torch.cuda.synchronize()
begin_time = time.time()

for i in range(100):
    pred = model(imgs, F10, F12)
    F10, F12 = pred[1], pred[2]

torch.cuda.synchronize()
end_time = time.time()

print('Average Runtime: {:.2f} ms'.format(
    (end_time - begin_time) / 100 / 2 * 1000))
