import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, firwin


class AudioEqualizer:
    def __init__(self, eq_params, mode="IIR"):
        """
        eq_params: dict with structure {
            'cutoff_low': float,
            'cutoff_high': float,
            'gain_low': float,
            'gain_mid': float,
            'gain_high': float
        }
        mode: 'IIR' or 'FIR'
        """
        self.cutoff_low = eq_params["cutoff_low"]
        self.cutoff_high = eq_params["cutoff_high"]
        self.gain_low = eq_params["gain_low"]
        self.gain_mid = eq_params["gain_mid"]
        self.gain_high = eq_params["gain_high"]
        self.mode = mode
        self.sample_rate = None

    def load_audio(self, path):
        """Load audio from path"""
        signal, sr = sf.read(path)
        if signal.ndim == 1:  # mono -> (N, 1)
            signal = signal[:, np.newaxis]
        self.sample_rate = sr
        return signal, sr

    def save_audio(self, signal, name_prefix, suffix):
        """Save audio (handles mono/stereo automatically)."""
        sf.write(f"results/{name_prefix}_{suffix}.wav", signal, self.sample_rate)

    def filter_band(self, signal, band):
        """Apply a band filter (IIR or FIR)."""
        nyq = 0.5 * self.sample_rate

        if self.mode == "IIR":
            if band == "low":
                b, a = butter(4, self.cutoff_low / nyq, btype="low")
            elif band == "mid":
                b, a = butter(
                    4, [self.cutoff_low / nyq, self.cutoff_high / nyq], btype="band"
                )
            else:  # high
                b, a = butter(4, self.cutoff_high / nyq, btype="high")

            filtered = lfilter(b, a, signal)

        elif self.mode == "FIR":
            numtaps = 513
            if band == "low":
                b = firwin(numtaps, self.cutoff_low / nyq, pass_zero="lowpass")
            elif band == "mid":
                b = firwin(
                    numtaps,
                    [self.cutoff_low / nyq, self.cutoff_high / nyq],
                    pass_zero="bandpass",
                )
            else:  # high
                b = firwin(numtaps, self.cutoff_high / nyq, pass_zero="highpass")
            filtered = np.convolve(signal, b, mode="same")
        else:
            raise ValueError("Invalid mode. Choose 'IIR' or 'FIR'.")

        return filtered

    def apply_equalization(self, audio_path):
        signal, _ = self.load_audio(audio_path)
        name_prefix = audio_path.split("/")[-1].split(".")[0]

        equalized = np.zeros_like(signal)

        # Process each channel
        for ch in range(signal.shape[1]):
            s = signal[:, ch]

            sig_low = self.filter_band(s, "low") * 10 ** (self.gain_low / 20)
            sig_mid = self.filter_band(s, "mid") * 10 ** (self.gain_mid / 20)
            sig_high = self.filter_band(s, "high") * 10 ** (self.gain_high / 20)

            out = sig_low + sig_mid + sig_high
            out = np.clip(out, -1.0, 1.0)
            equalized[:, ch] = out

        self.save_audio(signal, name_prefix, "original")
        self.save_audio(equalized, name_prefix, f"equalized_{self.mode}")

        self.plot_frequency_response(signal, equalized, name_prefix)
        print(f"Done! Files saved as results/{name_prefix}_*.wav")

    def plot_frequency_response(self, signal_original, signal_filtered, name_prefix):
        """Plot frequency spectra for both channels using subplots (Left/Right)."""
        n = signal_original.shape[0]
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)
        num_channels = signal_original.shape[1]

        plt.figure(figsize=(12, 5 * num_channels))

        for ch in range(num_channels):
            spec_orig = 20 * np.log10(
                np.abs(np.fft.rfft(signal_original[:, ch])) + 1e-6
            )
            spec_eq = 20 * np.log10(np.abs(np.fft.rfft(signal_filtered[:, ch])) + 1e-6)

            ax = plt.subplot(num_channels, 1, ch + 1)
            label_LR = f"Left" if ch == 0 else f"Right"
            ax.plot(freqs, spec_orig, label=f"Original {label_LR}", alpha=0.7)
            ax.plot(freqs, spec_eq, label=f"Equalized {label_LR}", alpha=0.7)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"Frequency Spectrum - {label_LR}")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"results/{name_prefix}_{self.mode}_spectrum.png", dpi=300)
        plt.close()

    def generate_test_signal(self, duration=3.0):
        """Generate stereo test signal with three sine tones."""
        sr = 44100
        self.sample_rate = sr
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        left = (
            0.6 * np.sin(2 * np.pi * 200 * t)  # Low
            + 0.4 * np.sin(2 * np.pi * 1000 * t)  # Mid
            + 0.3 * np.sin(2 * np.pi * 6000 * t)  # High
        )

        right = (
            0.6 * np.sin(2 * np.pi * 300 * t)
            + 0.4 * np.sin(2 * np.pi * 1200 * t)
            + 0.3 * np.sin(2 * np.pi * 8000 * t)
        )

        signal = np.stack([left, right], axis=1)
        signal = signal / np.max(np.abs(signal))
        sf.write("music/test_signal.wav", signal, sr)
        return signal
