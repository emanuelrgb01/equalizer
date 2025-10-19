import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class AudioEqualizer:
    """
    AudioEqualizer: handles FIR and IIR 3-band equalization on mono/stereo audio.
    eq_params can use either names 'cutoff_low'/'cutoff_high' and 'gain_low' etc.
    """

    def __init__(self, eq_params, mode="FIR"):
        # accept multiple possible key names to avoid KeyError
        def get_param(d, *keys, default=None):
            for k in keys:
                if k in d:
                    return d[k]
            return default

        self.cutoff_low = get_param(
            eq_params, "cutoff_low", "low_cutoff", default=300.0
        )
        self.cutoff_high = get_param(
            eq_params, "cutoff_high", "high_cutoff", default=3000.0
        )

        self.gain_low_db = get_param(eq_params, "gain_low", "low_gain", default=0.0)
        self.gain_mid_db = get_param(eq_params, "gain_mid", "mid_gain", default=0.0)
        self.gain_high_db = get_param(eq_params, "gain_high", "high_gain", default=0.0)

        self.gain_low = 10 ** (self.gain_low_db / 20.0)
        self.gain_mid = 10 ** (self.gain_mid_db / 20.0)
        self.gain_high = 10 ** (self.gain_high_db / 20.0)

        # HEADROOM
        max_boost_db = max(0.0, self.gain_low_db, self.gain_mid_db, self.gain_high_db)
        # Calculate an attenuation gain to create headroom and avoid clipping.
        # If the largest boost is +X dB, the headroom_gain will be -X dB (on a linear scale).
        # If there is no boost (only clipping), the gain is 1.0 (no attenuation).
        self.headroom_gain = 10 ** (-max_boost_db / 20.0)

        self.mode = mode.upper()
        if self.mode not in ("FIR", "IIR"):
            raise ValueError("mode must be 'FIR' or 'IIR'")

        # design params for FIR
        self.fir_taps = 1025

        # IIR bands
        self.iir_bands = [
            (self.cutoff_low, 0.7, self.gain_low_db),
            ((self.cutoff_high + self.cutoff_low) / 2, 1.0, self.gain_mid_db),
            (self.cutoff_high, 0.7, self.gain_high_db),
        ]

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

    def _peaking_eq(self, f0, Q, gain_db, fs):
        """RBJ peaking EQ coefficients (b, a)"""
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * Q)
        cosw0 = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cosw0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw0
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a

    def _process_channel_fir(self, x, sr):
        low_h, band_h, high_h = self._design_fir_bands(sr)
        sig_low = signal.fftconvolve(x, low_h, mode="same") * self.gain_low
        sig_mid = signal.fftconvolve(x, band_h, mode="same") * self.gain_mid
        sig_high = signal.fftconvolve(x, high_h, mode="same") * self.gain_high
        out = sig_low + sig_mid + sig_high

        # Applies headroom gain to avoid clipping, instead of using np.clip.
        out *= self.headroom_gain
        return out

    def _process_channel_iir(self, x, sr):
        # Use three peaking filters approximating low/mid/high.
        # For low/high we use peaking with low Q to emulate shelving-like behavior.
        bands = self.iir_bands

        y = x.copy()
        for f0, Q, gdb in bands:
            b, a = self._peaking_eq(f0, Q, gdb, sr)
            y = signal.lfilter(b, a, y)

        # Applies headroom gain to avoid clipping, instead of using np.clip.
        y *= self.headroom_gain
        return y

    def process(self, audio, sr):
        """
        audio: numpy array shape (N,) for mono or (N,2) for stereo
        returns processed audio same shape
        """
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

        plt.figure(figsize=(8, 4))
        if self.mode.upper() == "FIR":
            low_h, band_h, high_h = self._design_fir_bands(sr)
            H_low = np.fft.rfft(low_h * self.gain_low, n=N)
            H_mid = np.fft.rfft(band_h * self.gain_mid, n=N)
            H_high = np.fft.rfft(high_h * self.gain_high, n=N)
            H_tot = H_low + H_mid + H_high

            # Applies headroom gain to correct plotting.
            H_tot *= self.headroom_gain

            mag_db = 20 * np.log10(np.maximum(np.abs(H_tot), 1e-12))

            freqs = np.fft.rfftfreq(N, 1 / sr)
            plt.title("Combined FIR Equalizer Response (with Headroom Gain)")
        else:
            # IIR mode
            H_tot = np.ones(N // 2, dtype=complex)

            for f0, Q, gdb in self.iir_bands:
                b, a = self._peaking_eq(f0, Q, gdb, sr)
                w, h = signal.freqz(b, a, worN=N // 2, fs=sr)
                H_tot *= h

            # Applies headroom gain to correct plotting.
            H_tot *= self.headroom_gain
            mag_db = 20 * np.log10(np.maximum(np.abs(H_tot), 1e-12))

            freqs = w
            plt.title("Combined IIR Equalizer Response (with Headroom Gain)")

        plt.semilogx(freqs, mag_db)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.axhline(0, color="red", linestyle=":", linewidth=0.8, label="0 dB")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{audio_name}_{self.mode}_filter.png", dpi=200)
        plt.close()
