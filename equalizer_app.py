import os
import threading
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from audio_equalizer import AudioEqualizer


class EqualizerGUI:
    def __init__(self, root):
        self.root = root
        root.title("3-Band Equalizer")
        root.geometry("520x620")

        self.filepath = None
        self.mode = tk.StringVar(value="FIR")

        self.gain_low = tk.DoubleVar(value=0.0)
        self.gain_mid = tk.DoubleVar(value=0.0)
        self.gain_high = tk.DoubleVar(value=0.0)

        self.cutoff_low = tk.DoubleVar(value=300.0)
        self.cutoff_high = tk.DoubleVar(value=3000.0)

        tk.Label(
            root, text="3-Band Digital Equalizer", font=("Arial", 14, "bold")
        ).pack(pady=8)

        file_frame = tk.Frame(root)
        file_frame.pack(pady=4)
        tk.Button(
            file_frame, text="Select audio (wav/mp3)", command=self.select_file
        ).pack(side="left", padx=6)
        self.file_label = tk.Label(file_frame, text="No file selected", fg="gray")
        self.file_label.pack(side="left")

        mode_frame = tk.Frame(root)
        mode_frame.pack(pady=6)
        tk.Label(mode_frame, text="Mode:").pack(side="left", padx=(0, 6))
        tk.Radiobutton(mode_frame, text="FIR", variable=self.mode, value="FIR").pack(
            side="left"
        )
        tk.Radiobutton(mode_frame, text="IIR", variable=self.mode, value="IIR").pack(
            side="left"
        )

        # sliders + freq ranges
        self._add_slider(root, "Low (Bass) dB", self.gain_low, -24, 24)
        tk.Label(root, textvariable=self.cutoff_low, fg="blue").pack()
        tk.Label(root, text=f"Low range: 20 Hz - {self.cutoff_low.get()} Hz").pack()

        self._add_slider(root, "Mid (Middle) dB", self.gain_mid, -24, 24)
        tk.Label(
            root,
            text=f"Mid range: {self.cutoff_low.get()} Hz - {self.cutoff_high.get()} Hz",
        ).pack()

        self._add_slider(root, "High (Treble) dB", self.gain_high, -24, 24)
        tk.Label(root, textvariable=self.cutoff_high, fg="blue").pack()
        tk.Label(
            root, text=f"High range: {self.cutoff_high.get()} Hz - nyquist_freq Hz"
        ).pack()

        # buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Apply EQ", command=self.on_apply).pack(
            side="left", padx=6
        )
        tk.Button(btn_frame, text="Play Original", command=self.play_original).pack(
            side="left", padx=6
        )
        tk.Button(btn_frame, text="Play Equalized", command=self.play_equalized).pack(
            side="left", padx=6
        )

        self.status_label = tk.Label(root, text="", fg="green")
        self.status_label.pack(pady=6)

        # last processed path
        self.last_equalized_path = None
        self.audio_data = None
        self.sample_rate = None

    def _add_slider(self, parent, label, var, lo, hi):
        frame = tk.Frame(parent)
        frame.pack(pady=4, fill="x")
        tk.Label(frame, text=label).pack(anchor="w")
        tk.Scale(
            frame,
            from_=lo,
            to=hi,
            orient="horizontal",
            resolution=0.5,
            variable=var,
            length=420,
        ).pack()

    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a"), ("All files", "*.*")]
        )
        if not path:
            return
        self.filepath = path
        self.file_label.config(text=os.path.basename(path), fg="black")

        try:
            data, sr = librosa.load(
                path, sr=None, mono=False
            )  # shape (n,) or (channels, n)
            # convert to shape (n, channels)
            if data.ndim == 1:
                audio = data[:, np.newaxis]
            else:
                audio = data.T  # librosa returns (channels, n)
            self.audio_data = audio
            self.sample_rate = sr
            self.status_label.config(
                text=f"Loaded {os.path.basename(path)} ({'stereo' if audio.shape[1]==2 else 'mono'}, {sr} Hz)"
            )
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load audio: {e}")
            self.audio_data = None
            self.sample_rate = None

    def on_apply(self):
        if not self.filepath or self.audio_data is None:
            messagebox.showwarning("No file", "Please select an audio file first.")
            return

        eq_params = {
            "cutoff_low": float(self.cutoff_low.get()),
            "cutoff_high": float(self.cutoff_high.get()),
            "gain_low": float(self.gain_low.get()),
            "gain_mid": float(self.gain_mid.get()),
            "gain_high": float(self.gain_high.get()),
        }
        mode = self.mode.get()

        # run processing in background
        threading.Thread(
            target=self._process_thread, args=(eq_params, mode), daemon=True
        ).start()

    def _process_thread(self, eq_params, mode):
        try:
            self.status_label.config(text="Processing...")
            eq = AudioEqualizer(eq_params, mode=mode)

            # audio is (n, channels)
            processed = eq.process(self.audio_data, self.sample_rate)
            name = os.path.splitext(os.path.basename(self.filepath))[0]
            out_path = f"results/{name}_{mode.lower()}_eq.wav"

            sf.write(out_path, processed, self.sample_rate)
            self.last_equalized_path = out_path
            # also save original into results for convenience
            orig_path = f"results/{name}_original.wav"
            sf.write(orig_path, self.audio_data, self.sample_rate)

            # save filter response plot (FIR or IIR)
            try:
                eq.plot_filter_responses(self.sample_rate, audio_name=name)
            except Exception as e:
                print(f"Warning: could not plot filter response ({e})")

            # plot comparative spectra (two subplots)
            self._plot_compare_spectra(self.audio_data, processed, name)

            self.status_label.config(text=f"Done — saved: {out_path}")
            messagebox.showinfo("Done", f"Equalized audio saved: {out_path}")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            messagebox.showerror("Processing error", str(e))

    def _plot_compare_spectra(self, orig, proc, name_prefix):
        n = orig.shape[0]
        freqs = np.fft.rfftfreq(n, 1 / self.sample_rate)

        num_channels = orig.shape[1]
        fig, axs = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels))
        if num_channels == 1:
            axs = [axs]

        for ch in range(num_channels):
            spec_orig = 20 * np.log10(np.abs(np.fft.rfft(orig[:, ch])) + 1e-12)
            spec_proc = 20 * np.log10(np.abs(np.fft.rfft(proc[:, ch])) + 1e-12)
            ax = axs[ch]
            ax.semilogx(freqs, spec_orig, label="Original", alpha=0.7)
            ax.semilogx(freqs, spec_proc, label="Equalized", alpha=0.7)
            label_name = "Left" if ch == 0 else "Right"
            ax.set_title(f"Channel: {label_name}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, which="both", ls="--", lw=0.4)
            ax.legend()

        mode = self.mode.get()
        plt.tight_layout()
        plt.savefig(f"results/{name_prefix}_{mode}_spectrum_compare.png", dpi=200)
        plt.close(fig)

    def _play_audio_array(self, arr, sr):
        """
        Play a numpy array audio (shape (n,) or (n,2)). Runs in background thread.
        Uses sounddevice; if unavailable, shows error.
        """
        try:
            sd.play(arr, sr)
            sd.wait()
        except Exception as e:
            messagebox.showerror("Playback error", f"Playback failed: {e}")

    def play_original(self):
        if self.audio_data is None:
            messagebox.showwarning("No audio", "No audio loaded.")
            return
        # play in background thread
        threading.Thread(
            target=self._play_audio_array,
            args=(self.audio_data, self.sample_rate),
            daemon=True,
        ).start()

    def play_equalized(self):
        if not self.last_equalized_path:
            messagebox.showwarning(
                "No equalized audio", "No equalized audio yet. Apply EQ first."
            )
            return
        try:
            data, sr = librosa.load(self.last_equalized_path, sr=None, mono=False)
            if data.ndim == 1:
                arr = data
            else:
                arr = data.T  # (n, ch)
            threading.Thread(
                target=self._play_audio_array, args=(arr, sr), daemon=True
            ).start()
        except Exception as e:
            messagebox.showerror(
                "Playback error", f"Could not load equalized audio for playback: {e}"
            )
