from ResNet import resnet18 as resnet
from dataloader import Res_Dataset
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.funcs import load_json
from tqdm import tqdm


def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)

seed = 5
device = torch.device('cuda')

cfg = load_json('config/base.json')
train_dataset = Res_Dataset(cfg)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=16,
                                               pin_memory=True,
                                               drop_last=True,
                                               worker_init_fn=train_dataset.worker_init_fn
                                           )


model = resnet(pretrained=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = nn.CrossEntropyLoss()


iter_loss = []
train_losses = []
train_accs = []
min_acc = 0.0
for epoch in range(100):
    np.random.seed(seed + epoch)
    train_loss = 0.
    train_acc = 0.
    for batch_idx, data in tqdm(enumerate(train_loader), total=100, desc="Processing"):
        img = data['img'].to(device, non_blocking=True)
        target = data['label'].to(device, non_blocking=True)
        output = model(img)
        loss = criterion(output, target)
        loss_value = loss.item()
        iter_loss.append(loss_value)
        train_loss += loss_value
        acc = compute_accuray(F.log_softmax(output, dim=1), target)
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_acc / len(train_loader))
    print("Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
            epoch + 1, 100, train_loss / len(train_loader), train_acc / len(train_loader),))
    if (train_acc / len(train_loader))>min_acc:
        save_path = './outputs/'+str(epoch)+'_'+str(train_acc / len(train_loader))+'_val.tar'
        torch.save(
            {
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }, save_path
        )
