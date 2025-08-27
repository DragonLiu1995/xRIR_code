import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]
    

def stft_l1_loss(gt_spect=None, pred_spect=None, mask=None, logspace=True, log1p_gt=False,
                 log_instead_of_log1p_in_logspace=True,  log_gt=False, log_gt_eps=1.0e-8, mag_only=True,):
    """
    compute L1 loss between gt and estimated spectrograms (spect.)
    :param gt_spect: gt spect.
    :param pred_spect: estimated spect.
    :param mask: mask to mark valid entries in batch
    :param logspace: flag to tell if spect. estimation in log-space or not
    :param log1p_gt: flag to decide to log(1 + gt_spect) or not
    :param log_instead_of_log1p_in_logspace: flag to decide to log(gt_spect/pred_spect) instead of log(1 + gt_spect/pred_spect)
    :param log_gt: flag to decide to log(gt_spect)
    :param log_gt_eps: eps to be added before computing log for numerical stability
    :param mag_only: flag is set if spect. is magnitude only
    :return: L1 loss between gt and estimated spects.
    """
    if mag_only:
        assert torch.all(gt_spect >= 0.).item(), "mag_only"

        if logspace:
            if log_instead_of_log1p_in_logspace:
                if log_gt:
                    gt_spect = torch.log(gt_spect + log_gt_eps)
                else:
                    pred_spect = torch.exp(pred_spect) - log_gt_eps
            else:
                if log1p_gt:
                    gt_spect = torch.log1p(gt_spect)
                else:
                    pred_spect = torch.exp(pred_spect) - 1

        if mask is not None:
            assert mask.size()[:1] == gt_spect.size()[:1] == pred_spect.size()[:1]
            # pred_spect, gt_spect: B x H x W x C; mask: B
            gt_spect = gt_spect * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            pred_spect = pred_spect * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError

    if mask is None:
        loss = F.l1_loss(pred_spect, gt_spect)
    else:
        # not counting the contribution from masked out locations in the batch
        loss = torch.sum(torch.abs(pred_spect - gt_spect)) / (torch.sum(mask) * np.prod(list(pred_spect.size())[1:]))

    return loss


def compute_spect_energy_decay_losses(loss_type="l1_loss",
                                      loss_weight=1.0e-2,
                                      gts=None,
                                      preds=None,
                                      mask=None,
                                      slice_till_direct_signal=False,
                                      direct_signal_len_in_ms=50,
                                      dont_collapse_across_freq_dim=False,
                                      sr=16000,
                                      hop_length=62,
                                      win_length=248,
                                      ):
    """
    compute energy decay loss
    :param loss_type: loss type
    :param loss_weight: loss weight
    :param gts: gt IRs
    :param preds: estimated IRs
    :param mask: mask to mark valid entries in batch
    :param slice_till_direct_signal: remove direct signal part of IR
    :param direct_signal_len_in_ms: direct signal length in milliseconds
    :param dont_collapse_across_freq_dim: collapse along frequency dimension of spectrogram (spect.)
    :param sr: sampling rate
    :param hop_length: hop length to compute spect.
    :param win_length: length of temporal window to compute spect.
    :return: energy decay loss
    """
    assert len(gts.size()) == len(preds.size()) == 4
    assert gts.size(-1) in [1, 2]
    assert preds.size(-1) in [1, 2]
    print(gts.shape, preds.shape)

    slice_idx = None
    if slice_till_direct_signal:
        if direct_signal_len_in_ms == 50:
            if (sr == 16000) and (hop_length == 62) and (win_length == 248):
                # (62 * 11 + 248 / 2) / 16000 = 0.050375 (50 ms)
                # so has to use the 12th window (idx = 11).. so slice idx should be 12
                slice_idx = 12
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    if slice_till_direct_signal:
        if dont_collapse_across_freq_dim:
            gts_fullBandAmpEnv = gts[:slice_idx]
        else:
            gts_fullBandAmpEnv = torch.sum(gts[:slice_idx], dim=-3)
    else:
        if dont_collapse_across_freq_dim:
            gts_fullBandAmpEnv = gts
        else:
            gts_fullBandAmpEnv = torch.sum(gts**2, dim=-3)
    power_gts_fullBandAmpEnv = gts_fullBandAmpEnv #** 2
    energy_gts_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_gts_fullBandAmpEnv, [-2]), -2), [-2])
    valid_loss_idxs = ((energy_gts_fullBandAmpEnv != 0.).type(energy_gts_fullBandAmpEnv.dtype))[..., 1:, :]

    db_gts_fullBandAmpEnv = 10 * torch.log10(energy_gts_fullBandAmpEnv + 1.0e-13)
    norm_db_gts_fullBandAmpEnv = db_gts_fullBandAmpEnv - db_gts_fullBandAmpEnv[..., :1, :]
    norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv[..., 1:, :]
    if slice_till_direct_signal:
        weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv
    else:
        weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv * valid_loss_idxs

    if slice_till_direct_signal:
        if dont_collapse_across_freq_dim:
            preds_fullBandAmpEnv = preds[:slice_idx]
        else:
            preds_fullBandAmpEnv = torch.sum(preds[:slice_idx], dim=-3)
    else:
        if dont_collapse_across_freq_dim:
            preds_fullBandAmpEnv = preds
        else:
            preds_fullBandAmpEnv = torch.sum(preds**2, dim=-3)
    power_preds_fullBandAmpEnv = preds_fullBandAmpEnv #** 2
    energy_preds_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_preds_fullBandAmpEnv, [-2]), -2), [-2])
    db_preds_fullBandAmpEnv = 10 * torch.log10(energy_preds_fullBandAmpEnv + 1.0e-13)
    norm_db_preds_fullBandAmpEnv = db_preds_fullBandAmpEnv - db_preds_fullBandAmpEnv[..., :1, :]
    norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv[..., 1:, :]
    if slice_till_direct_signal:
        weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv
    else:
        weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv * valid_loss_idxs

    if loss_type == "l1_loss":
        if mask is None:
            loss = F.l1_loss(weighted_norm_db_preds_fullBandAmpEnv, weighted_norm_db_gts_fullBandAmpEnv)
        else:
            # not counting the contribution from masked out locations in the batch
            assert torch.sum(mask) == mask.size(0)
            loss = torch.sum(torch.abs(weighted_norm_db_preds_fullBandAmpEnv - weighted_norm_db_gts_fullBandAmpEnv)) /\
                   (torch.sum(mask) * np.prod(list(weighted_norm_db_preds_fullBandAmpEnv.size())[1:]))
    else:
        raise NotImplementedError

    loss = loss * loss_weight

    return loss



def compute_energy_db(h):
    h = np.array(h)
    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    return  energy_db


class Criterion(torch.nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, h, target_h):
        h = h.squeeze(1)
        target_h = target_h.squeeze(1)
        # Compute energy decay in dB for the network output h
        power = h ** 2
        energy = torch.cumsum(power.flip(dims=[-1]), dim=-1).flip(dims=[-1])

        # Mask out zero tails for each batch independently
        non_zero_mask = (energy > 0)
        energy_db_list = []
        for i in range(energy.size(0)):
            i_nz = non_zero_mask[i].nonzero(as_tuple=True)[0].max().item() if non_zero_mask[i].any() else energy.size(-1) - 1
            energy_trimmed = energy[i, :i_nz+1]
            energy_db = 10 * torch.log10(energy_trimmed + 1e-10)
            energy_db = energy_db.clone() - energy_db[0]  # Normalize to start at 0 dB
            energy_db_list.append(energy_db)

        # Pad the energy_db_list to the maximum length for batch consistency
        energy_db = torch.nn.utils.rnn.pad_sequence(energy_db_list, batch_first=True)
        energy_db = energy_db[...,:8000]
        # Compute energy decay in dB for the target impulse response target_h
        target_power = target_h ** 2
        target_energy = torch.cumsum(target_power.flip(dims=[-1]), dim=-1).flip(dims=[-1])

        # Repeat the same trimming and padding for target_h
        target_non_zero_mask = (target_energy > 0)
        target_energy_db_list = []
        for i in range(target_energy.size(0)):
            target_i_nz = target_non_zero_mask[i].nonzero(as_tuple=True)[0].max().item() if target_non_zero_mask[i].any() else target_energy.size(-1) - 1
            target_energy_trimmed = target_energy[i, :target_i_nz+1]
            target_energy_db = 10 * torch.log10(target_energy_trimmed + 1e-10)
            target_energy_db = target_energy_db.clone() - target_energy_db[0]  # Normalize to start at 0 dB
            target_energy_db_list.append(target_energy_db)

        target_energy_db = torch.nn.utils.rnn.pad_sequence(target_energy_db_list, batch_first=True)
        target_energy_db = target_energy_db[...,:8000]
        # Compute L1 loss over the padded sequences
        loss = torch.nn.functional.l1_loss(energy_db, target_energy_db)
        print("HHH", torch.nn.functional.l1_loss(energy_db[0], target_energy_db[0]))
        return loss


def train(model, train_loader, optimizer, lr_scheduler, epoch):
    model.train()
    log_interval = 1
    total_train_loss = 0.
    count_batch = 0
    for batch_idx, data in enumerate(train_loader):
        (
            proj_listener_pos, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos
        ) = data
        optimizer.zero_grad()
        out_spec, tgt_spec = model(depth_coord.cuda(), all_ref_irs.cuda(), proj_source_pos.cuda(),  all_ref_src_pos.cuda(), tgt_wav.cuda())

        decay_loss = compute_spect_energy_decay_losses(gts=tgt_spec.permute(0, 2, 3, 1).cuda(), preds=torch.exp(out_spec.cuda()) - 1e-8)
        stft_loss = stft_l1_loss(pred_spect=out_spec.cuda(), gt_spect=tgt_spec.permute(0, 2, 3, 1).cuda())
        log_stft_loss = F.l1_loss(out_spec.cuda(), torch.log(tgt_spec + 1e-8).permute(0, 2, 3, 1).cuda())
        loss = stft_loss + decay_loss #+ log_stft_loss
        print("HHH", stft_loss, decay_loss)#, log_stft_loss)

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        total_train_loss += loss.item()
        count_batch += 1
    lr_scheduler.step()
    print("Train Epoch: {}, Avg Train Loss: {}".format(epoch, total_train_loss / count_batch))


def test(model, test_loader, epoch):
    model.eval()
    test_loss = 0
    avg_acc = 0.
    cnt = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            (
            proj_listener_pos, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos
        ) = data
            out_spec, tgt_spec = model(depth_coord.cuda(), all_ref_irs.cuda(), proj_source_pos.cuda(),  all_ref_src_pos.cuda(), tgt_wav.cuda())
            decay_loss = compute_spect_energy_decay_losses(gts=tgt_spec.permute(0, 2, 3, 1).cuda(), preds=torch.exp(out_spec.cuda()) - 1e-8)
            stft_loss = stft_l1_loss(pred_spect=out_spec.cuda(), gt_spect=tgt_spec.permute(0, 2, 3, 1).cuda())
            # log_stft_loss = F.l1_loss(out_spec.cuda(), torch.log(tgt_spec + 1e-8).permute(0, 2, 3, 1).cuda())
            loss = stft_loss + decay_loss #+ log_stft_loss
            print("HHH", stft_loss, decay_loss)#, log_stft_loss)
            test_loss += loss.item()
            cnt += 1
    avg_acc = avg_acc / cnt
    test_loss = test_loss / cnt
    print('\nTest set: Average loss: {:.4f}, Avg Acc: {}'.format(
        test_loss, avg_acc))
    return test_loss


if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set the device
        torch.cuda.set_device(0) 

        # Verify the current device
        print(torch.cuda.current_device())
    from sim_to_real.haa_4_rooms_dataset import HAA_All_Dataset
    from torch.utils.data import DataLoader
    from model.xRIR import xRIR
    import torch.optim as optim
    lr = 1e-4
    decay_epochs = 50
    lr_gamma = 0.1
    weight_decay = 0.0001


    num_epoch = 1000
    train_dataset = HAA_All_Dataset(split="train", max_len=9600, num_shot=8)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)

    test_dataset = HAA_All_Dataset(split="test",  max_len=9600, num_shot=8)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=128)


    model = xRIR(condition_dim=6, num_channels=8, input_dim=1, latent_dim=256)
    model.load_state_dict(torch.load("/path/to/your/best/ckpt_trained_on_acousticrooms/xx.pth", map_location="cpu"))
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

    for ep in range(0, num_epoch + 1):
        train(model, train_loader, optimizer, lr_scheduler, ep)
        test_loss = test(model, test_loader, ep)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "./path/to/saved_ckpt_folder/{}.pth".format(ep)) 

    
