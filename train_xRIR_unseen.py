import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
from utils.spec_utils import stft_l1_loss, compute_spect_energy_decay_losses


def train(model, train_loader, optimizer, lr_scheduler, epoch):
    model.train()
    log_interval = 1
    total_train_loss = 0.
    count_batch = 0
    for batch_idx, data in enumerate(train_loader):
        (
            _, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos
        ) = data
        optimizer.zero_grad()
        out_spec, tgt_spec = model(depth_coord.cuda(), all_ref_irs.cuda(), proj_source_pos.cuda(),  all_ref_src_pos.cuda(), tgt_wav.cuda())

        decay_loss = compute_spect_energy_decay_losses(gts=tgt_spec.permute(0, 2, 3, 1).cuda(), preds=torch.exp(out_spec.cuda()) - 1e-8)
        stft_loss = stft_l1_loss(pred_spect=out_spec.cuda(), gt_spect=tgt_spec.permute(0, 2, 3, 1).cuda())
        loss = stft_loss + decay_loss

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(proj_source_pos), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        total_train_loss += loss.item()
        count_batch += 1
    lr_scheduler.step()
    print("Train Epoch: {}, Avg Train Loss: {}".format(epoch, total_train_loss / count_batch))


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    cnt = 0
    with torch.no_grad():
        for data in test_loader:
            (
            _, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos
        ) = data
            out_spec, tgt_spec = model(depth_coord.cuda(), all_ref_irs.cuda(), proj_source_pos.cuda(),  all_ref_src_pos.cuda(), tgt_wav.cuda())
            decay_loss = compute_spect_energy_decay_losses(gts=tgt_spec.permute(0, 2, 3, 1).cuda(), preds=torch.exp(out_spec.cuda()) - 1e-8)
            stft_loss = stft_l1_loss(pred_spect=out_spec.cuda(), gt_spect=tgt_spec.permute(0, 2, 3, 1).cuda())
            loss = stft_loss + decay_loss
            test_loss += loss.item()
            cnt += 1
    test_loss = test_loss / cnt
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == "__main__":
    # Check if CUDA is available
    # if torch.cuda.is_available():
    #     # Set the device
    #     torch.cuda.set_device(0) 

    #     # Verify the current device
    #     print(torch.cuda.current_device())
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    from torch.utils.data import DataLoader
    from model.xRIR import xRIR
    from utils.lr_scheduler import ExponentialLR
    import torch.optim as optim
    lr = 1e-3
    decay_epochs = 50
    lr_gamma = 0.1
    weight_decay = 0.0001
    batch_size = 64
    num_shot = 8 # 1, 4, 8


    num_epoch = 200
    train_dataset = xRIR_Dataset(split="train", max_len=9600, num_shot=8)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    test_dataset = xRIR_Dataset(split="test",  max_len=9600, num_shot=8)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


    model = xRIR(num_channels=num_shot)
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = ExponentialLR(optimizer, decay_epochs=decay_epochs,
                                 gamma=lr_gamma)


    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    
    best_test_loss = 1e9

    for ep in range(1, num_epoch + 1):
        train(model, train_loader, optimizer, lr_scheduler, ep)
        test_loss = test(model, test_loader, ep)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "./ckpt/xRIR_{}_shot/{}_best.pth".format(num_shot, ep))
        else:
            torch.save(model.state_dict(), "./ckpt/xRIR_{}_shot/{}_loss_{}.pth".format(num_shot, ep, round(float(test_loss), 3)))
