import numpy as np
import torch.optim
from torch.utils.data import DataLoader
import random
from utils.dataloader import dataloader
from utils.loss_function import loss_function
from utils.logs import log
from datetime import datetime
import os
from utils.funcs import load_json
from net.aeNet.conv_autoencoder_pixelshuffle import conv_autoencoder
from net.resnet.ResNet import Detector as ResNet
from torch.utils.tensorboard import SummaryWriter
import cv2
writer = SummaryWriter("logs")
from tqdm import tqdm


def main():
    cfg = load_json("config/base.json")
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    now = datetime.now()
    save_path = os.path.join('output', 'train_' + now.strftime("%m_%d_%H_%M_%S"))
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'weights'))
    os.mkdir(os.path.join(save_path, 'logs'))
    logger = log(path=os.path.join(save_path, "logs"), file="losses.logs")

    # TODO
    train_dataset = dataloader(cfg=cfg["dataloader"], n_file=cfg["num_train"], n_frames=cfg["num_frames"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg["batch_size"],
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True)

    device = torch.device('cuda')
    model = conv_autoencoder().to(device)

    # resnet
    resnet = ResNet().to(device)
    res_tmp = torch.load('net/resnet/checkpoints/resnet34.tar')["model"]
    resnet.load_state_dict(res_tmp)
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    weight = []
    n_weight = 5
    batch_counter = 0
    for epoch in range(cfg["epoch"]):
        for batch_idx, (real, fake, additional_data) in \
                tqdm(enumerate(train_loader), total=cfg["num_train"] // (cfg["batch_size"] / cfg["num_frames"]),
                     desc="Processing"):
            real, fake = real.to(device), fake.to(device)
            M1, M2, M3 = model(fake)
            FF_loss, AD_loss, QA_loss, PI_loss = loss_function(M1, M2, M3, real, fake, resnet, additional_data, cfg)
            loss = FF_loss + AD_loss + QA_loss + PI_loss
            writer.add_scalar('loss', loss, batch_counter)
            writer.add_scalar('FF_loss', FF_loss, batch_counter)
            writer.add_scalar('AD_loss', AD_loss, batch_counter)
            writer.add_scalar('QA_loss', QA_loss, batch_counter)
            writer.add_scalar('PI_loss', PI_loss, batch_counter)
            batch_counter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        log_text = "epoch:{} lr:{} FF_loss:{:.5f} AD_loss:{:.5f} QA_loss:{:.5f} PI_loss:{:.5f} loss:{:.5f}".format(
            epoch, optimizer.state_dict()['param_groups'][0]['lr'], FF_loss, AD_loss, QA_loss, PI_loss, loss)

        if len(weight) < n_weight:
            save_model_path = os.path.join(save_path, 'weights', "{}_{:.4f}_train.tar".format(epoch, loss))
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss
            }, save_model_path)
            weight.append([loss, "{}_{:.4f}_train.tar".format(epoch, loss)])
            weight.sort(reverse=True)
        elif loss < weight[0][0]:
            save_model_path = os.path.join(save_path, 'weights', "{}_{:.4f}_train.tar".format(epoch, loss))
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss
            }, save_model_path)
            weight.append([loss, "{}_{:.4f}_train.tar".format(epoch, loss)])
            weight.sort(reverse=True)
            os.remove(os.path.join(save_path, 'weights', weight[0][1]))
            del weight[0]

        logger.info(log_text)
    writer.close()


if __name__ == '__main__':
    main()
