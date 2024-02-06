from model import get_model
import torch.nn.functional as F
from dataloader import get_train_loader

loader = get_train_loader()
model = get_model().to('cuda')

for b in loader:
    b = b.float().to('cuda')
    output = model(b).to('cuda')
    print(F.binary_cross_entropy(F.sigmoid(output), b))
    print(F.mse_loss(output, b))
