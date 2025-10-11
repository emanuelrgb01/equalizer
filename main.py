import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


def peaking_eq(f0, Q, gain_db, fs):
    """Design a peaking EQ (RBJ Audio EQ Cookbook)."""
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


def equalize_audio(input_path: str, mode: str, eq_params: dict):
    """
    Apply a 3-band audio equalizer (FIR or IIR) to an audio file.
    Supports mono and stereo audio.
    """

    # Load audio using librosa (float32)
    signal_in, sample_rate = librosa.load(input_path, sr=None, mono=False)

    if signal_in.ndim == 1:
        signal_in = signal_in[np.newaxis, :]  # shape (1, n_samples)
    n_channels, n_samples = (
        signal_in.shape if signal_in.shape[0] <= 2 else signal_in.shape[::-1]
    )

    # Extract audio name from path
    audio_name = input_path.split("/")[-1].split(".")[0]
    print(
        f"Loaded: {input_path}, Channels: {signal_in.shape[0]}, Sample rate: {sample_rate} Hz"
    )

    # Extract EQ parameters
    f1 = eq_params.get("cutoff_low", 300.0)
    f2 = eq_params.get("cutoff_high", 3000.0)
    gain_db_low = eq_params.get("gain_low", 0.0)
    gain_db_mid = eq_params.get("gain_mid", 0.0)
    gain_db_high = eq_params.get("gain_high", 0.0)

    gain_low = 10 ** (gain_db_low / 20.0)
    gain_mid = 10 ** (gain_db_mid / 20.0)
    gain_high = 10 ** (gain_db_high / 20.0)

    # Define equalizer
    def process_channel(sig_channel):
        if mode.upper() == "FIR":
            nyq = sample_rate / 2.0
            numtaps = 1025
            low = signal.firwin(
                numtaps, cutoff=f1 / nyq, pass_zero="lowpass", window="hamming"
            )
            band = signal.firwin(
                numtaps, [f1 / nyq, f2 / nyq], pass_zero="bandpass", window="hamming"
            )
            high = signal.firwin(
                numtaps, cutoff=f2 / nyq, pass_zero="highpass", window="hamming"
            )

            sig_low = signal.fftconvolve(sig_channel, low, mode="same") * gain_low
            sig_mid = signal.fftconvolve(sig_channel, band, mode="same") * gain_mid
            sig_high = signal.fftconvolve(sig_channel, high, mode="same") * gain_high

            out = sig_low + sig_mid + sig_high
            out = out / np.max(np.abs(out))
        elif mode.upper() == "IIR":
            s = sig_channel.copy()
            bands = [
                (100.0, 0.7, gain_db_low),
                (1000.0, 1.0, gain_db_mid),
                (8000.0, 0.7, gain_db_high),
            ]
            for f0, Q, gdb in bands:
                b, a = peaking_eq(f0, Q, gdb, sample_rate)
                s = signal.lfilter(b, a, s)
            out = s / np.max(np.abs(s))
        else:
            raise ValueError("Mode must be 'FIR' or 'IIR'.")
        return out

    # Spectrum comparison
    def plot_spectrum_comparison(sig_orig, sig_eq, sr, channel_idx, audio_name, mode):
        N = len(sig_orig)
        freq = np.fft.rfftfreq(N, 1 / sr)
        spec_orig = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(sig_orig)), 1e-12))
        spec_eq = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(sig_eq)), 1e-12))

        plt.figure(figsize=(10, 5))
        plt.semilogx(freq, spec_orig, label="Original")
        plt.semilogx(freq, spec_eq, label=f"Equalized ({mode})")
        plt.title(f"Spectrum Comparison - Channel {channel_idx+1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"results/{audio_name}_channel{channel_idx+1}_{mode.lower()}_spectrum.png"
        )
        plt.close()

    # Process all channels
    output_channels = []
    for ch in range(signal_in.shape[0]):
        out_ch = process_channel(signal_in[ch])
        output_channels.append(out_ch)
    signal_out = np.vstack(output_channels)  # shape: (channels, samples)

    for ch in range(n_channels):
        plot_spectrum_comparison(
            signal_in[ch], signal_out[ch], sample_rate, ch, audio_name, mode
        )
    # Save outputs
    sf.write(
        f"results/{audio_name}_original.wav",
        signal_in.T.astype(np.float32),
        sample_rate,
    )
    sf.write(
        f"results/{audio_name}_{mode.lower()}_eq.wav",
        signal_out.T.astype(np.float32),
        sample_rate,
    )

    print("\nProcessing complete!")
    print(f"Original saved as: results/{audio_name}_original.wav")
    print(f"Equalized saved as: results/{audio_name}_{mode.lower()}_eq.wav")


## Usage
eq_params = {
    "cutoff_low": 300.0,
    "cutoff_high": 3000.0,
    "gain_low": -6.0,
    "gain_mid": -3.0,
    "gain_high": +25.0,
}

# equalize_audio("music/from_the_start.wav", mode="FIR", eq_params=eq_params)
# equalize_audio("music/from_the_start.wav", mode="IIR", eq_params=eq_params)
# equalize_audio("music/thunderstruck.mp3", mode="FIR", eq_params=eq_params)
# equalize_audio("music/thunderstruck.mp3", mode="IIR", eq_params=eq_params)
# equalize_audio("music/hotel_california.mp3", mode="FIR", eq_params=eq_params)
# equalize_audio("music/hotel_california.mp3", mode="IIR", eq_params=eq_params)
