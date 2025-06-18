import sys
import numpy as np

import numpy as np
import os

from scipy import stats
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
import torchaudio
from scipy.signal import hilbert

def griffin_lim(spec, n_fft=124, win_length=62, hop_length=31, power=1, n_iter=32):
    transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, n_iter=n_iter,win_length=win_length,hop_length=hop_length,power=power
    )
    return transform(spec)


def normalize(audio, norm='peak'):
    """
    normalize IR
    :param audio: IR
    :param norm: normalization mode
    :return: normalized IR
    """
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError


def measure_drr_energy_ratio(y, cutoff_time=0.003, fs=22050):
    """
    get direct to reverberant energy ratio (DRR)
    :param y: IR
    :param cutoff_time: cutoff time to compute DRR
    :param fs: sampling frequency
    :return: DRR
    """
    direct_sound_idx = int(cutoff_time * fs)

    # removing leading silence
    y = normalize(y)
    y = np.trim_zeros(y, trim='fb')

    # everything up to the given idx is summed up and treated as direct sound energy
    y = np.power(y, 2)
    direct = sum(y[:direct_sound_idx + 1])
    reverberant = sum(y[direct_sound_idx + 1:])
    if direct == 0 or reverberant == 0:
        drr = 1
        # print('Direct or reverberant is 0')
    else:
        drr = 10 * np.log10(direct / reverberant)

    return drr


def calculate_drr_diff(gt, est, cutoff_time=0.003, fs=22050, compute_relative_diff=False, get_diff_val=True,
                       get_gt_val=False, get_pred_val=False,):
    """
    get difference in DRR, DRR for gt or DRR for estimated IR
    :param gt: gt IR
    :param est: estimated IR
    :param cutoff_time: cutoff time to compute DRR
    :param fs: sampling frequency
    :param compute_relative_diff: flag to compute relative difference
    :param get_diff_val: flag to get difference in DRR
    :param get_gt_val: flag to get DRR of gt IR
    :param get_pred_val: flag to get DRR of estimated IR
    :return: difference in DRR, DRR of gt IR or DRR of estimated IR
    """
    drr_gt = measure_drr_energy_ratio(gt, cutoff_time=cutoff_time, fs=fs)
    drr_est = measure_drr_energy_ratio(est, cutoff_time=cutoff_time, fs=fs)
    diff = abs(drr_gt - drr_est)
    if compute_relative_diff:
        diff = abs(diff / drr_gt)

    if get_diff_val:
        return diff
    elif get_gt_val:
        return drr_gt
    elif get_pred_val:
        return drr_est



class Evaluator:
    def __init__(self, cfg=None, seq_name=None):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.cfg = cfg
        self.seq_name = seq_name
        self.t60_error = []
        self.clarity_error = []
        self.edt_error = []
        self.spec_mse = []

        self.invalid = 0

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def stft_loss(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        return mse

    def env_loss(self, pred_wav, gt_wav):
        pred_env = np.abs(hilbert(pred_wav))
        gt_env = np.abs(hilbert(gt_wav))
        envelope_distance = np.mean(np.abs(gt_env - pred_env) / np.max(gt_env)) * 100.
        return float(envelope_distance)
    
    def t60_impulse(self, energy, rt='t20', fs=22050, trans=False, trans1=False):
        rt = rt.lower()
        if rt == 't30':
            init = -5.0
            end = -35.0
            factor = 2.0
        elif rt == 't20':
            init = -5.0
            end = -25.0
            factor = 3.0
        elif rt == 't10':
            init = -5.0
            end = -15.0
            factor = 6.0
        elif rt == 'edt':
            init = 0.0
            end = -10.0
            factor = 6.0

        if trans:
            result = energy[0][..., None]
            new_energy = []
            for band in range(result.shape[1]):
                # Filtering signal
                filtered_signal = result[:, band]
                abs_signal = np.abs(filtered_signal)
                # Schroeder integration
                sch = abs_signal**2
                new_energy.append(sch)
            energy = np.array(new_energy)
        elif trans1:
            abs_signal = np.abs(energy[0])
            energy = np.array(abs_signal**2).reshape(1, -1, 22).mean(-1)
            factor *= 1
            trans = False
        else:
            factor *= 1

        t60 = np.zeros(energy.shape[0])
        c50 = np.zeros(energy.shape[0])
        schdb_list = []
        for band in range(energy.shape[0]):
            if trans:
                pow_energy = energy[band]
            else:
                if not trans1:
                    pow_energy = np.power(10, energy[band])
                else:
                    pow_energy = energy[band]
                x_dense = np.arange(0, pow_energy.shape[-1]*22)
                x_bin = np.arange(0, pow_energy.shape[-1]*22, 22)
                f = interp1d(x_bin, pow_energy, kind = 'slinear')
                pow_energy = f(x_dense[:len(x_bin)*22-22])
                # pow_energy = np.interp(x_dense, x_bin, pow_energy)
            sch = np.cumsum(pow_energy[::-1])[::-1]
            sch_db = 10.0 * np.log10(sch / np.max(sch))
            sch_db -= sch_db[0]
            if band == 0:
                out_sch_db = sch_db
            schdb_list.append(sch_db)

            init_sample = np.min(np.where(-5 - sch_db > 0)[0])
            if len(np.where(-35 - sch_db> 0)[0]) > 0:
                end_sample = np.min(np.where(-35 - sch_db> 0)[0])
            else:
                end_sample = len(sch_db) - 1
            t60[band] = factor * (end_sample / fs - init_sample / fs)
            if trans1 or not trans:
                t = int((50 / 1000.0) * fs + 1)
            else:
                t = int((50 / 1000.0) * fs + 1)
            c50[band] = 10.0 * np.log10((np.sum(pow_energy[:t]) / np.sum(pow_energy[t:])))
        return t60, out_sch_db, c50, init_sample
    
    def measure_edt(self, h, fs=22050, decay_db=10):
        h = np.array(h)
        fs = float(fs)

        # The power of the impulse response in dB
        power = h ** 2
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]

        i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
        t_decay = i_decay / fs
        # compute the decay time
        decay_time = t_decay
        est_edt = (60 / decay_db) * decay_time
        return est_edt

    def measure_rt60(self, h, fs=22050, decay_db=20, plot=False, rt60_tgt=None):
        """
        Analyze the RT60 of an impulse response. Optionaly plots some useful information.
        Parameters
        ----------
        h: array_like
            The impulse response.
        fs: float or int, optional
            The sampling frequency of h (default to 1, i.e., samples).
        decay_db: float or int, optional
            The decay in decibels for which we actually estimate the time. Although
            we would like to estimate the RT60, it might not be practical. Instead,
            we measure the RT20 or RT30 and extrapolate to RT60.
        plot: bool, optional
            If set to ``True``, the power decay and different estimated values will
            be plotted (default False).
        rt60_tgt: float
            This parameter can be used to indicate a target RT60 to which we want
            to compare the estimated value.
        """

        h = np.array(h)
        fs = float(fs)

        # The power of the impulse response in dB
        power = h ** 2
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]
        # -5 dB headroom
        i_5db = np.min(np.where(-5 - energy_db > 0)[0])
        e_5db = energy_db[i_5db]
        t_5db = i_5db / fs

        i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
        t_decay = i_decay / fs

        # compute the decay time
        decay_time = t_decay - t_5db
        est_rt60 = (60 / decay_db) * decay_time

        return est_rt60
    def measure_clarity(self, signal, time=50, fs=22050):
        h2 = signal**2
        t = int((time/1000)*fs + 1)
        return 10*np.log10(np.sum(h2[:t])/np.sum(h2[t:]))
    def compute_energy_db(self, h):
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
    def evaluate_edt(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir
        np_gt_ir = gt_ir
        pred_edt = self.measure_edt(np_pred_ir)
        gt_edt = self.measure_edt(np_gt_ir)
        edt_error = abs(pred_edt - gt_edt)
        self.edt_error.append(edt_error)

    def evaluate_clarity(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir
        np_gt_ir = gt_ir
        pred_clarity = self.measure_clarity(np_pred_ir)
        gt_clarity = self.measure_clarity(np_gt_ir)
        clarity_error = abs(pred_clarity - gt_clarity)
        self.clarity_error.append(clarity_error)

    def evaluate_t60(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir
        np_gt_ir = gt_ir
        mse = np.mean((np_pred_ir - np_gt_ir) ** 2)
        self.mse.append(mse)
        psnr = self.psnr_metric(np_pred_ir, np_gt_ir)
        self.psnr.append(psnr)
        try:
            pred_t60 = self.measure_rt60(np_pred_ir)
            gt_t60 = self.measure_rt60(np_gt_ir)
            t60_error = abs(pred_t60 - gt_t60) / gt_t60
            self.t60_error.append(t60_error)
        except:
            self.invalid += 1
    def evaluate_energy_db(self, pred_ir, gt_ir):
        pred_db = self.compute_energy_db(pred_ir)
        gt_db = self.compute_energy_db(gt_ir)
        return pred_db, gt_db
    def evaluate_spec_mse(self, pred_ir_spec, gt_ir_spec):
        self.spec_mse.append(np.mean(pred_ir_spec - gt_ir_spec)**2)
   

def apply_delay(signal, delay_tensor):
    """
    Apply integer delay to each sample in the batch with zero-padding.
    
    Args:
    - signal: Tensor of shape [batch_size, sequence_length, channels] or [batch_size, sequence_length].
    - delay_tensor: Tensor of shape [batch_size] with integer delay values for each sample.
    
    Returns:
    - Delayed signal with the same shape as input signal.
    """
    batch_size, sequence_length = signal.shape[:2]
    
    # Output tensor initialized with zeros (same shape as input signal)
    delayed_signal = torch.zeros_like(signal)
    
    for i in range(batch_size):
        delay = delay_tensor[i].item()  # Get the delay value for this batch item
        
        if delay > 0:
            # Positive delay: shift right and pad with zeros at the beginning
            delayed_signal[i, delay:] = signal[i, :-delay]
        elif delay < 0:
            # Negative delay: shift left and pad with zeros at the end
            delayed_signal[i, :delay] = signal[i, -delay:]
        else:
            # No delay, just copy the signal
            delayed_signal[i] = signal[i]
    
    return delayed_signal

if __name__ == "__main__":

    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    from torch.utils.data import DataLoader
    from model.xRIR import xRIR
    import torch
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import librosa


    # Check if CUDA is available
    if torch.cuda.is_available():
        # Set the device
        torch.cuda.set_device(0) 

        # Verify the current device
        print(torch.cuda.current_device())


    test_dataset = xRIR_Dataset(num_shot = 8, split="test")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1)


    model = xRIR(condition_dim=6, num_channels=8, input_dim=1, latent_dim=256)
    model.load_state_dict(torch.load("./checkpoints/xRIR_unseen.pth", map_location="cpu"))
    model.cuda()

    model.eval()
    test_loss = 0
    cnt = 0
    t60_error_list = []
    c50_error_list = []
    edt_error_list = []
    drre_list = []
    mag_loss_list = []
    env_loss_list = []
    count_outlier = 0
    with torch.no_grad():
        for data in test_loader:
            proj_listener_pos, proj_source_pos, depth, tgt_wav, all_ref_irs, all_ref_src_pos = data

            out_spec, tgt_spec =  model(depth.cuda(), all_ref_irs.cuda(), proj_source_pos.cuda(),  all_ref_src_pos.cuda(), tgt_wav.cuda())
            
            pred_mag_spec = torch.exp(out_spec) - 1e-8
            pred_mag_spec = pred_mag_spec[...,0].cpu()
            out_wav = griffin_lim(pred_mag_spec).unsqueeze(0)


            evaluator = Evaluator()
            gt_edt = evaluator.measure_edt(tgt_wav[0, 0, :8000].cpu().numpy())
            pred_edt = evaluator.measure_edt(out_wav[0, 0, :8000].cpu().numpy())
            print("GT EDT: {}s, Pred EDT: {}s, EDT Error: {}s".format(gt_edt, pred_edt, np.abs(gt_edt - pred_edt)))
            edt_error_list.append(np.abs(gt_edt - pred_edt))


            gt_c50 = evaluator.measure_clarity(tgt_wav[0, 0, :8000].cpu().numpy())
            pred_c50 = evaluator.measure_clarity(out_wav[0, 0, :8000].cpu().numpy())
            print("GT C50: {}dB, Pred C50: {}dB, C50 Error: {}dB".format(gt_c50, pred_c50, np.abs(gt_c50 - pred_c50)))
            if np.abs(gt_c50 - pred_c50) != np.inf and np.abs(gt_c50 - pred_c50) != -np.inf:
                c50_error_list.append(np.abs(gt_c50 - pred_c50))
            else:
                count_outlier += 1

            gt_t60 = evaluator.measure_rt60(tgt_wav[0, 0, :8000].cpu().numpy())
            pred_t60 = evaluator.measure_rt60(out_wav[0, 0, :8000].cpu().numpy())
            print("GT T60: {}s, Pred T60: {}s, T60 Percentage Error: {}\%".format(gt_t60, pred_t60, np.abs(gt_t60 - pred_t60) / gt_t60 * 100.))
            t60_error_list.append(np.abs(gt_t60 - pred_t60) / gt_t60 * 100.)

            mag_loss = evaluator.stft_loss(out_spec.squeeze(-1).cpu().numpy(), torch.log(tgt_spec + 1e-8).squeeze(1).cpu().numpy())
            print("STFT loss: ".format(mag_loss))
            mag_loss_list.append(mag_loss)
            print("HHH", cnt)

            cnt += 1
            print("Average EDT error: ", np.mean(edt_error_list))
            print("Average C50 error: ", np.mean(c50_error_list))
            print("Average T60 error: ", np.mean(t60_error_list))
            print("----------{}----------".format(cnt))

        print("Average EDT error: ", np.mean(edt_error_list))
        print("Average C50 error: ", np.mean(c50_error_list))
        print("Average T60 error: ", np.mean(t60_error_list))
        print("Number of Outliers: ", count_outlier)
