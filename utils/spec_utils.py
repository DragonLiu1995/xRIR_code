import numpy as np
from pyroomacoustics.experimental.rt60 import measure_rt60
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import auraloss


def griffin_lim(spec, n_fft=511, win_length=248, hop_length=62, power=1, n_iter=32):
    transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, n_iter=n_iter,win_length=win_length,hop_length=hop_length,power=power
    )
    return transform(spec)


class WaveL2Loss(nn.Module):
    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(WaveL2Loss, self).__init__()

    def forward(self, x_wav, y_wav):
        return (x_wav - y_wav).pow(2).mean()
    

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (Tensor): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    window = window.to(x.device)
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False, pad_mode='constant')
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initialize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of ground truth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize los STFT magnitude loss module."""
        super(STFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(y_mag, x_mag)


class LogMagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(LogMagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        if len(x.shape) == 3:
            loss = self.compute_one_channel(x[:, 0], y[:, 0]) + self.compute_one_channel(x[:, 1], y[:, 1])
        else:
            loss = self.compute_one_channel(x, y)

        return loss

    def compute_one_channel(self, x, y):
        window = self.window.to(x.device)
        mag_x = torch.stft(x, self.fft_size, self.shift_size, self.win_length, window, return_complex=True,
                           pad_mode='constant').abs()
        mag_y = torch.stft(y, self.fft_size, self.shift_size, self.win_length, window, return_complex=True,
                           pad_mode='constant').abs()
        loss = F.mse_loss(torch.log1p(mag_x), torch.log1p(mag_y))

        return loss


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.factor = fft_size / 2048

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss * self.factor, mag_loss * self.factor


class MagLogMagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(MagLogMagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.stft_magnitude_loss = STFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        log_mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        mag_loss = self.stft_magnitude_loss(x_mag, y_mag)

        return log_mag_loss + mag_loss


class MagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(MagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)
        self.stft_magnitude_loss = STFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        mag_loss = self.stft_magnitude_loss(x_mag, y_mag)

        return mag_loss


class RelativeMagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(RelativeMagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)
        self.stft_magnitude_loss = STFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag_l = stft(x[:, 0], self.fft_size, self.shift_size, self.win_length, self.window)
        x_mag_r = stft(x[:, 1], self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag_l = stft(y[:, 0], self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag_r = stft(y[:, 1], self.fft_size, self.shift_size, self.win_length, self.window)
        mag_loss = self.stft_magnitude_loss(x_mag_l - x_mag_r, y_mag_l - y_mag_r)

        return mag_loss


class MagRelativeMagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(MagRelativeMagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)
        self.stft_magnitude_loss = STFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag_l = stft(x[:, 0], self.fft_size, self.shift_size, self.win_length, self.window)
        x_mag_r = stft(x[:, 1], self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag_l = stft(y[:, 0], self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag_r = stft(y[:, 1], self.fft_size, self.shift_size, self.win_length, self.window)
        x_diff = stft(x[:, 0] - x[:, 1], self.fft_size, self.shift_size, self.win_length, self.window)
        y_diff = stft(y[:, 0] - y[:, 1], self.fft_size, self.shift_size, self.win_length, self.window)
        mag_loss = self.stft_magnitude_loss(x_diff, y_diff)
        mag_loss_l = self.stft_magnitude_loss(x_mag_l, y_mag_l)
        mag_loss_r = self.stft_magnitude_loss(x_mag_r, y_mag_r)

        return mag_loss + mag_loss_l + mag_loss_r


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths, factor_sc, factor_mag):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, 'hamming_window')]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss


class MultiResolutionNoSCSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionNoSCSTFTLoss, self).__init__()

        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [MagLogMagSTFTLoss(fs, ss, wl, 'hamming_window')]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        mag_loss = 0.0
        for f in self.stft_losses:
            mag_l = f(x, y)
            mag_loss += mag_l
        mag_loss /= len(self.stft_losses)

        return mag_loss



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


def compute_spect_losses(loss_types=[],
                         loss_weights=[],
                         gt_spect=None,
                         pred_spect=None,
                         mask=None,
                         logspace=True,
                         log1p_gt=False,
                         log_instead_of_log1p_in_logspace=True,
                         log_gt=False,
                         log_gt_eps=1.0e-8,):
    """
    get spectrogram (spect.) loss (error in spect. estimation)
    :param loss_types: loss type
    :param loss_weights: loss weight
    :param gt_spect: gt IR spect.
    :param pred_spect: estimated IR spec.
    :param mask: mask to mark valid entries in batch
    :param logspace: flag to tell if spect. estimation in log-space or not
    :param log1p_gt: flag to decide to log(1 + gt_spect) or not
    :param log_instead_of_log1p_in_logspace: flag to decide to log(gt_spect/pred_spect) instead of log(1 + gt_spect/pred_spect)
    :param log_gt: flag to decide to log(gt_spect)
    :param log_gt_eps: eps to be added before computing log for numerical stability
    :return: spect. loss
    """
    loss = 0.
    for loss_idx, loss_type in enumerate(loss_types):
        if loss_type == "stft_l1_loss":
            loss += (stft_l1_loss(
                gt_spect=gt_spect,
                pred_spect=pred_spect,
                mask=mask,
                logspace=logspace,
                log1p_gt=log1p_gt,
                log_instead_of_log1p_in_logspace=log_instead_of_log1p_in_logspace,
                log_gt=log_gt,
                log_gt_eps=log_gt_eps,
            ) * loss_weights[loss_idx])
        else:
            raise ValueError

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



def get_focal_loss(pred, target, alpha=0.25, gamma=2, reduction='mean'):
    label = target.long()
    pt = (1 - pred) * label + pred * (1 - label)
    focal_weight = (alpha * label + (1 - alpha) * (1 - label)) * pt.pow(gamma)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,
        target, reduction='none') * focal_weight
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss.mean()


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mse_loss = torch.nn.MSELoss()#lambda x, y : torch.sum((x - y) ** 2)
        # self.l1_loss = torch.nn.SmoothL1Loss(reduction='sum', beta=1e-2)
        self.mse_loss = torch.nn.MSELoss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1.0, w_phs=0.0,
                                                               fft_sizes=[192, 48, 24, 12],
                                                               hop_sizes=[96, 24, 12, 6],
                                                               win_lengths=[192, 48, 24, 12],
                                                            )
        # self.bce_loss = torch.nn.BCELoss()
        # self.fig = plt.figure()
        # self.fig_c = self.fig.add_subplot(1, 1, 1)
    def forward(self, pred_ir, gt_ir):
        scalar_stats = {}
        wave_l2 = self.mse_loss(pred_ir, gt_ir)
        spec_loss = self.mrft_loss(pred_ir, gt_ir)
        mse_loss = wave_l2 * 5000 + spec_loss
        # print("HHH", wave_l2 * 5000, spec_loss)
        # scalar_stats['mse_loss'] = mse_loss
        return mse_loss