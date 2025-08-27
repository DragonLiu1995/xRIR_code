import sys
sys.path.append("/pscratch/sd/x/xiuliu/room_rir")
import numpy as np

import numpy as np
import os

from scipy import stats
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interp1d
import torchaudio
from scipy.signal import hilbert

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
        self.fig = plt.figure()
        self.figplot = self.fig.add_subplot(2, 1, 1)
        self.figplot1 = self.fig.add_subplot(2, 1, 2)

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

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
            # octave = OctaveBandsFactory(base_frequency=125, fs=fs, n_fft=512)
            # result = octave.analysis(energy[0])
            result = energy[0][..., None]
            # abs_signal = np.abs(energy[0])
            # energy = np.array(abs_signal**2).reshape(1, -1)

            new_energy = []
            for band in range(result.shape[1]):
                # Filtering signal
                filtered_signal = result[:, band]
                abs_signal = np.abs(filtered_signal)
                # Schroeder integration
                sch = abs_signal**2
                new_energy.append(sch)
            energy = np.array(new_energy)
            # energy = energy.reshape(8, -1, 44).mean(-1)
        elif trans1:
            abs_signal = np.abs(energy[0])
            energy = np.array(abs_signal**2).reshape(1, -1, 22).mean(-1)
            factor *= 1
            trans = False
        else:
            # energy = np.sqrt(np.power(10, energy)).sum(0).reshape(1, -1)
            # trans = True
            factor *= 1

        t60 = np.zeros(energy.shape[0])
        c50 = np.zeros(energy.shape[0])
        schdb_list = []
        for band in range(energy.shape[0]):
            if trans:
                pow_energy = energy[band]
            else:
                if not trans1:
                    # mean = -5.99
                    # std = 4.85
                    # energy[band] = energy[band] * std + mean

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
            #     # import pdb; pdb.set_trace()
                out_sch_db = sch_db
            schdb_list.append(sch_db)
            
            # sch_init = sch_db[np.abs(sch_db - init).argmin()]
            # sch_end = sch_db[np.abs(sch_db - end).argmin()]

            # if len(np.where(sch_db == sch_init)[0])==0 or len(np.where(sch_db == sch_end)[0])==0:
            #     return [0.0] * 8
            # init_sample = np.where(sch_db == sch_init)[0][0]
            # end_sample = np.where(sch_db == sch_end)[0][0]

            init_sample = np.min(np.where(-5 - sch_db > 0)[0])
            if len(np.where(-35 - sch_db> 0)[0]) > 0:
                end_sample = np.min(np.where(-35 - sch_db> 0)[0])
            else:
                end_sample = len(sch_db) - 1
            # x = np.arange(init_sample, end_sample + 1) / fs
            # # x = np.arange(init_sample, end_sample + 1)
            # y = sch_db[init_sample:end_sample + 1]
            # if len(x)==0:
            #     return [0.0] * 8
            # slope, intercept = stats.linregress(x, y)[0:2]
            # # slope = round(slope, 1)
            # # intercept = round(intercept, 1)
            # db_regress_init = (init - intercept) / slope
            # db_regress_end = (end - intercept) / slope
            # t60[band] = factor * (db_regress_end - db_regress_init)
            t60[band] = factor * (end_sample / fs - init_sample / fs)
            if trans1 or not trans:
                # t = int((50 / 1000.0 / 44.0) * fs + 1)
                t = int((50 / 1000.0) * fs + 1)
            else:
                # t = int((50 / 1000.0 / 44.0) * fs + 1)
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

    def measure_rt60(self, h, fs=22050, decay_db=30, plot=False, rt60_tgt=None):
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

        # after decay
        # if len(np.where(-5-decay_db - energy_db >0)[0]) == 0:
        #     return 100
        i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
        t_decay = i_decay / fs

        # compute the decay time
        decay_time = t_decay - t_5db
        est_rt60 = (60 / decay_db) * decay_time
        # c50 = 10.0 * np.log10((np.sum(pow_energy[:t]) / np.sum(pow_energy[t:])))
        if plot:
            import matplotlib.pyplot as plt

            # Remove clip power below to minimum energy (for plotting purpose mostly)
            energy_min = energy[-1]
            energy_db_min = energy_db[-1]
            power[power < energy[-1]] = energy_min
            power_db = 10 * np.log10(power)
            power_db -= np.max(power_db)

            # time vector
            def get_time(x, fs):
                return np.arange(x.shape[0]) / fs - i_5db / fs

            T = get_time(power_db, fs)

            # plot power and energy
            plt.plot(get_time(energy_db, fs), energy_db, label="Energy")

            # now the linear fit
            plt.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
            plt.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
            plt.vlines(
                est_rt60, energy_db_min, 0, linestyles="dashed", label="Estimated RT60"
            )

            if rt60_tgt is not None:
                plt.vlines(rt60_tgt, energy_db_min, 0, label="Target RT60")

            plt.legend()

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
        np_pred_ir = pred_ir#.data.cpu().numpy()
        np_gt_ir = gt_ir#.data.cpu().numpy()
        mse = np.mean((np_pred_ir - np_gt_ir) ** 2)
        self.mse.append(mse)
        psnr = self.psnr_metric(np_pred_ir, np_gt_ir)
        self.psnr.append(psnr)
        try:
            pred_t60 = self.measure_rt60(np_pred_ir)
            gt_t60 = self.measure_rt60(np_gt_ir)
            t60_error = abs(pred_t60 - gt_t60) / gt_t60
            self.t60_error.append(t60_error)
            # plt.figure()
            # plt.plot(pred_energy, c='b')
            # plt.plot(gt_energy, c='r')
            # plt.savefig('/home/ksu/kun_naf/results/db.jpg')
                # plt.ylim(-1, 1)
        except:
            self.invalid += 1
    def evaluate_energy_db(self, pred_ir, gt_ir):
        np_pred_ir = pred_ir#.data.cpu().numpy()
        np_gt_ir = gt_ir#.data.cpu().numpy()
        pred_db = self.compute_energy_db(pred_ir)
        gt_db = self.compute_energy_db(gt_ir)
        return pred_db, gt_db
    def evaluate_spec_mse(self, pred_ir_spec, gt_ir_spec):
        self.spec_mse.append(np.mean(pred_ir_spec - gt_ir_spec)**2)
    def evaluate(self, pred_ir, gt_ir, energy_hist, band_sch, name=None):
        pred_ir = pred_ir[:, :gt_ir.shape[-1]].data.cpu().numpy()
        gt_ir = gt_ir.data.cpu().numpy()
        mse = np.mean((pred_ir - gt_ir)**2)
        self.mse.append(mse)
        psnr = self.psnr_metric(pred_ir, gt_ir)
        self.psnr.append(psnr)
        gt_ir = gt_ir[...,:3894] #230*22
        pred_t60, pred_sch_db, pred_c50, pred_init_sample = self.t60_impulse(energy_hist.data.cpu().numpy())
        # gt_sample_t60, gt_sample_sch_db, gt_sample_c50 = self.t60_impulse(band_sch.data.cpu().numpy(), trans1=True)
        gt_sample_t60, gt_sample_sch_db, gt_sample_c50, gt_sample_init_sample = self.t60_impulse(band_sch.data.cpu().numpy())
        gt_t60, gt_sch_db, gt_c50, gt_init_sample = self.t60_impulse(gt_ir, trans=True)

        # print(f'gt c50: {gt_c50.mean()}')
        # print(f'gt_sample_c50: {gt_sample_c50.mean()}')
        # print(f'pred_c50: {pred_c50.mean()}')


        # self.figplot.plot([44*i for i in range(int(0.4*len(pred_sch_db)))], pred_sch_db[:int(0.4*len(pred_sch_db))], label='pred_db')
        # self.figplot.plot([44*i for i in range(int(0.4*len(gt_sample_sch_db)))], gt_sample_sch_db[:int(0.4*len(gt_sample_sch_db))], label='gt_sample_db')
        # self.figplot.plot(range(int(0.4*len(gt_sch_db))), gt_sch_db[:int(0.4*len(gt_sch_db))], label='gt_db')
        # self.figplot.plot([i for i in range(int(len(gt_sample_sch_db)))], gt_sample_sch_db[:int(len(gt_sample_sch_db))], label='gt_sample_db')
        fs = 22050
        self.figplot.plot(np.array([pred_init_sample, pred_t60*fs+pred_init_sample], dtype=float), np.array([pred_sch_db[pred_init_sample], -65], dtype=float), label='pred_db')
        self.figplot.plot(np.array([gt_init_sample, gt_t60*fs+gt_init_sample], dtype=float), np.array([gt_sch_db[gt_init_sample], -65], dtype=float), label='gt_db')

        self.figplot1.plot([i for i in range(int(len(pred_sch_db)))], pred_sch_db[:int(len(pred_sch_db))], label='pred_db')
        self.figplot1.plot([i for i in range(int(len(gt_sample_sch_db)))], gt_sample_sch_db[:int(len(gt_sample_sch_db))], label='gt_sample_db')
        self.figplot1.plot(range(int(len(gt_sch_db))), gt_sch_db[:int(len(gt_sch_db))], label='gt_db')
        self.figplot1.set_ylim(-80, 0)
        # import pdb; pdb.set_trace()
        # plt.scatter(range(8), pred_t60, label='pred_t60')
        # plt.scatter(range(8), gt_sample_t60, label='gt_sample_t60')
        # plt.scatter(range(8), gt_t60, label='gt_t60')
        self.figplot1.legend(loc='upper right')
        # self.fig.savefig('data/t60.jpg')
        self.fig.savefig('data/small2_t60.jpg')
        if name is not None:
            self.fig.savefig(name)
        self.figplot.cla()
        self.figplot1.cla()

        sample_t60_error = (abs(gt_sample_t60 - gt_t60) / gt_t60).mean() * 100
        # import pdb; pdb.set_trace()
        # sample_t60_error = (abs(pred_t60 - gt_sample_t60) / gt_sample_t60).mean() * 100
        # print(f'gt sample t60 error (avg percentage): {sample_t60_error}')

        t60_error = (abs(pred_t60 - gt_t60) / gt_t60).mean() * 100
        # print(f't60 error (avg percentage): {t60_error}')

        pred_edt, pred_sch_db, _, _ = self.t60_impulse(energy_hist.data.cpu().numpy(), rt='edt')
        # gt_sample_edt, gt_sample_sch_db, _ = self.t60_impulse(band_sch.data.cpu().numpy(), rt='edt', trans1=True)
        gt_sample_edt, gt_sample_sch_db, _, _  = self.t60_impulse(band_sch.data.cpu().numpy(), rt='edt')
        gt_edt, gt_sch_db, _, _ = self.t60_impulse(gt_ir, rt='edt', trans=True)

        sample_edt_error = (abs(gt_sample_edt - gt_edt)).mean()
        # print(f'gt sample edt error (avg percentage): {sample_edt_error}')

        edt_error = (abs(pred_edt - gt_edt)).mean()
        # print(f'edt error (avg percentage): {edt_error}')

        self.t60_error.append(t60_error)
        self.sample_t60_error.append(sample_t60_error)
        self.edt_error.append(edt_error)
        self.sample_edt_error.append(sample_edt_error)
        self.sample_c50_error.append(abs(gt_c50.mean()-gt_sample_c50.mean()))
        self.c50_error.append(abs(gt_c50.mean()-pred_c50.mean()))
    def simple_summarize(self):
        result_path = os.path.join(self.cfg.result_dir,
                                   self.seq_name, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        mse = np.mean(self.mse)
        psnr = np.mean(self.psnr)
        t60_error = np.mean(self.t60_error)
        clarity_error = np.mean(self.clarity_error)
        edt_error = np.mean(self.edt_error)
        spec_mse = np.mean(self.spec_mse)
        metrics = {'mse': mse, 'psnr': psnr, 'spec_mse': spec_mse, 't60_error': t60_error,
                   'clarity_error': clarity_error, 'edt_error': edt_error}
        return metrics
    def summarize(self):
        result_path = os.path.join(self.cfg.result_dir,
            self.seq_name, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        mse = np.mean(self.mse)
        psnr = np.mean(self.psnr)
        ssim = np.mean(self.ssim)
        t60_error = np.mean(self.t60_error)
        sample_t60_error = np.mean(self.sample_t60_error)
        edt_error = np.mean(self.edt_error)
        sample_edt_error = np.mean(self.sample_edt_error)
        c50_error = np.mean(self.c50_error)
        sample_c50_error = np.mean(self.sample_c50_error)

        metrics = {'mse': mse, 'psnr': psnr, 't60_error': t60_error, 'sample_t60_error': sample_t60_error,
            'edt_error': edt_error, 'sample_edt_error': sample_edt_error, 'c50_error': c50_error, 'sample_c50_error': sample_c50_error}

        # np.save(result_path, self.mse)
        # print('mse: {}'.format(mse))
        # print('psnr: {}'.format(psnr))
        self.mse = []
        self.psnr = []
        self.ssim = []
        return metrics

def griffin_lim(spec, n_fft=124, win_length=62, hop_length=31, power=1, n_iter=32):
    transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, n_iter=n_iter,win_length=win_length,hop_length=hop_length,power=power
    )
    return transform(spec)


if __name__ == "__main__":
    from eval_diff_rir_classroom.cond_depth_ir_few_shot_dataset_with_aug import Position_to_IR_Dataset
    from torch.utils.data import DataLoader
    # from model.cond_depth_few_shot import Cond_Depth_FewShot_IR_Net
    from model.few_shot_ir_interpolation_4 import MultiChannelDilatedConv1DAutoencoderWithPerChannelCondition, xRIR
    import torch
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import librosa


    test_dataset = Position_to_IR_Dataset(num_shot = 8, max_len=9600, split="test")
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)


    # model = MultiChannelDilatedConv1DAutoencoderWithPerChannelCondition(condition_dim=6, num_channels=8, input_dim=1, latent_dim=256)
    model = xRIR(condition_dim=6, num_channels=8, input_dim=1, latent_dim=256)
    # model.load_state_dict(torch.load("./ckpt/final_4_cond_ir_8_shot_pos_interp_multi_room_treble/4.pth", map_location="cpu")) # 77 48 final_4_cond_ir_8_shot_pos_interp_multi_room_treble
    model.load_state_dict(torch.load("./ckpt/xRIR_8_shot/12.pth", map_location="cpu")) #../treble_multi_room_exp/ckpt/xRIR/20_best.pth
    # model.load_state_dict(torch.load("../treble_multi_room_exp/ckpt/xRIR/20_best.pth"))
    # model.load_state_dict(torch.load("/pscratch/sd/x/xiuliu/room_rir/finetune_treble_collect_real/ckpt/xRIR_8_shot_scratch/19.pth", map_location="cpu")) #treble_multi_room_exp/ckpt/xRIR/20_best.pth /pscratch/sd/x/xiuliu/room_rir/treble_multi_room_exp

    model.cuda()

    resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=48000)


    # model = Cond_Depth_FewShot_IR_Net(n_bins=9600)
    # model.load_state_dict(torch.load("/pscratch/sd/x/xiuliu/room_rir/treble_multi_room_exp/ckpt/cond_depth_8_shot_multi_room_treble/73.pth")) # 52
    # model.cuda()

    model.eval()
    test_loss = 0
    cnt = 0
    t60_error_list = []
    c50_error_list = []
    edt_error_list = []
    env_loss_list = []
    drre_list = []
    with torch.no_grad():
        for data in test_loader:
            proj_listener_pos, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos = data

            out_spec, tgt_spec =  model(depth_coord.cuda(), all_ref_irs.cuda(), proj_source_pos.cuda(),  all_ref_src_pos.cuda(), tgt_wav.cuda())
            
            pred_mag_spec = torch.exp(out_spec) - 1e-8
            pred_mag_spec = pred_mag_spec[...,0].cpu()
            out_wav = griffin_lim(pred_mag_spec).unsqueeze(0) 

            min_len = min(out_wav.shape[-1], tgt_wav.shape[-1])
            evaluator = Evaluator()
            gt_edt = evaluator.measure_edt(tgt_wav[0, 0].cpu().numpy())
            pred_edt = evaluator.measure_edt(out_wav[0, 0].cpu().numpy())
            print("GT EDT: {}s, Pred EDT: {}s, EDT Error: {}s".format(gt_edt, pred_edt, np.abs(gt_edt - pred_edt)))
            edt_error_list.append(np.abs(gt_edt - pred_edt))


            gt_c50 = evaluator.measure_clarity(tgt_wav[0, 0].cpu().numpy())
            pred_c50 = evaluator.measure_clarity(out_wav[0, 0].cpu().numpy())
            print("GT C50: {}dB, Pred C50: {}dB, C50 Error: {}dB".format(gt_c50, pred_c50, np.abs(gt_c50 - pred_c50)))
            if np.abs(gt_c50 - pred_c50) != np.inf and np.abs(gt_c50 - pred_c50) != -np.inf:
                c50_error_list.append(np.abs(gt_c50 - pred_c50))

            gt_t60 = evaluator.measure_rt60(tgt_wav[0, 0].cpu().numpy())
            pred_t60 = evaluator.measure_rt60(out_wav[0, 0].cpu().numpy())
            print("GT T60: {}s, Pred T60: {}s, T60 Percentage Error: {}\%".format(gt_t60, pred_t60, np.abs(gt_t60 - pred_t60) / gt_t60 * 100.))
            t60_error_list.append(np.abs(gt_t60 - pred_t60) / gt_t60 * 100.)
            t60_err = np.abs(gt_t60 - pred_t60) / gt_t60 * 100.

            min_len = min(out_wav.shape[-1], tgt_wav.shape[-1])
            env_loss = evaluator.env_loss(out_wav[0, 0, :min_len].cpu().numpy(), tgt_wav[0, 0, :min_len].cpu().numpy())
            env_loss_list.append(env_loss)
            print("ENV loss: {}".format(env_loss))
            cnt += 1
            print("Average EDT error: ", np.mean(edt_error_list))
            print("Average C50 error: ", np.mean(c50_error_list))
            print("Average T60 error: ", np.mean(t60_error_list))
            print("Average env loss: ", np.mean(env_loss_list))
            if np.mean(c50_error_list) == np.inf:
                print(c50_error_list[-3:])
                exit(0)
        print("Average EDT error: ", np.mean(edt_error_list))
        print("Average C50 error: ", np.mean(c50_error_list))
        print("Average T60 error: ", np.mean(t60_error_list))
        print("Average env loss: ", np.mean(env_loss_list))