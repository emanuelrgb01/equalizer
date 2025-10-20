import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class AudioEqualizer:
    """
    AudioEqualizer: handles FIR and IIR 3-band equalization on mono/stereo audio.
    """

    def __init__(self, eq_params, mode="FIR"):
        def get_param(d, *keys, default=None):
            for k in keys:
                if k in d:
                    return d[k]
            return default

        self.cutoff_low = get_param(eq_params, "cutoff_low", default=300.0)
        self.cutoff_high = get_param(eq_params, "cutoff_high", default=3000.0)
        self.gain_low_db = get_param(eq_params, "gain_low", default=0.0)
        self.gain_mid_db = get_param(eq_params, "gain_mid", default=0.0)
        self.gain_high_db = get_param(eq_params, "gain_high", default=0.0)

        self.gain_low = 10 ** (self.gain_low_db / 20.0)
        self.gain_mid = 10 ** (self.gain_mid_db / 20.0)
        self.gain_high = 10 ** (self.gain_high_db / 20.0)

        max_boost_db = max(0.0, self.gain_low_db, self.gain_mid_db, self.gain_high_db)
        self.headroom_gain = 10 ** (-max_boost_db / 20.0)

        self.mode = mode.upper()
        if self.mode not in ("FIR", "IIR"):
            raise ValueError("mode must be 'FIR' or 'IIR'")

        self.fir_taps = 1025

    def _design_fir_bands(self, sr):
        nyq = sr / 2.0
        low = signal.firwin(
            self.fir_taps,
            cutoff=self.cutoff_low / nyq,
            pass_zero="lowpass",
            window="hamming",
        )
        band = signal.firwin(
            self.fir_taps,
            [self.cutoff_low / nyq, self.cutoff_high / nyq],
            pass_zero="bandpass",
            window="hamming",
        )
        high = signal.firwin(
            self.fir_taps,
            cutoff=self.cutoff_high / nyq,
            pass_zero="highpass",
            window="hamming",
        )
        return low, band, high

    def _process_channel_fir(self, x, sr):
        low_h, band_h, high_h = self._design_fir_bands(sr)
        sig_low = signal.fftconvolve(x, low_h, mode="same") * self.gain_low
        sig_mid = signal.fftconvolve(x, band_h, mode="same") * self.gain_mid
        sig_high = signal.fftconvolve(x, high_h, mode="same") * self.gain_high
        out = sig_low + sig_mid + sig_high

        out *= self.headroom_gain
        return out

    def _process_channel_iir(self, x, sr):
        """
        Process a single channel using a PARALLEL IIR filter bank.
        """

        # 4 Order Butterworth IIR filter
        order = 4
        nyq = 0.5 * sr

        # Low-pass
        b_lp, a_lp = signal.butter(order, self.cutoff_low / nyq, btype="lowpass")
        sig_low = signal.lfilter(b_lp, a_lp, x)

        # Band-pass
        b_bp, a_bp = signal.butter(
            order, [self.cutoff_low / nyq, self.cutoff_high / nyq], btype="bandpass"
        )
        sig_mid = signal.lfilter(b_bp, a_bp, x)

        # High-pass
        b_hp, a_hp = signal.butter(order, self.cutoff_high / nyq, btype="highpass")
        sig_high = signal.lfilter(b_hp, a_hp, x)

        y = (
            (sig_low * self.gain_low)
            + (sig_mid * self.gain_mid)
            + (sig_high * self.gain_high)
        )

        y *= self.headroom_gain
        return y

    def process(self, audio, sr):
        if audio.ndim == 1:
            # mono
            if self.mode == "FIR":
                return self._process_channel_fir(audio, sr)
            else:
                return self._process_channel_iir(audio, sr)
        else:
            # stereo: process each channel independently
            out = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                if self.mode == "FIR":
                    out[:, ch] = self._process_channel_fir(audio[:, ch], sr)
                else:
                    out[:, ch] = self._process_channel_iir(audio[:, ch], sr)
            return out

    def plot_filter_responses(self, sr, audio_name="audio"):
        """
        Plot combined filter magnitude responses (low+mid+high) including applied gains.
        Saves results/{audio_name}_{self.mode}_filter.png
        """
        N = 8192
        freqs = np.fft.rfftfreq(N, 1 / sr)
        plt.figure(figsize=(10, 5))

        if self.mode.upper() == "FIR":
            low_h, band_h, high_h = self._design_fir_bands(sr)
            H_low = np.fft.rfft(low_h * self.gain_low, n=N)
            H_mid = np.fft.rfft(band_h * self.gain_mid, n=N)
            H_high = np.fft.rfft(high_h * self.gain_high, n=N)
            H_tot = H_low + H_mid + H_high
            H_tot *= self.headroom_gain
            mag_db = 20 * np.log10(np.maximum(np.abs(H_tot), 1e-12))
            plt.title("Combined FIR Equalizer Response")
        else:
            order = 4
            nyq = 0.5 * sr

            b_lp, a_lp = signal.butter(order, self.cutoff_low / nyq, btype="lowpass")
            b_bp, a_bp = signal.butter(
                order, [self.cutoff_low / nyq, self.cutoff_high / nyq], btype="bandpass"
            )
            b_hp, a_hp = signal.butter(order, self.cutoff_high / nyq, btype="highpass")

            n_plot_points = N // 2 + 1

            w, H_low = signal.freqz(b_lp, a_lp, worN=n_plot_points)
            w, H_mid = signal.freqz(b_bp, a_bp, worN=n_plot_points)
            w, H_high = signal.freqz(b_hp, a_hp, worN=n_plot_points)

            H_tot = (
                (H_low * self.gain_low)
                + (H_mid * self.gain_mid)
                + (H_high * self.gain_high)
            )

            H_tot *= self.headroom_gain
            mag_db = 20 * np.log10(np.maximum(np.abs(H_tot), 1e-12))
            plt.title("Combined IIR Equalizer Response (Parallel Butterworth)")

        plt.semilogx(freqs, mag_db)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.axhline(0, color="red", linestyle=":", linewidth=0.8, label="0 dB")
        plt.ylim(-30, 30)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{audio_name}_{self.mode}_filter.png", dpi=200)
        plt.close()
