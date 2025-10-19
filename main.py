from audio_equalizer import AudioEqualizer

eq_params = {
    "cutoff_low": 400,
    "cutoff_high": 4000,
    "gain_low": 0,
    "gain_mid": -6,
    "gain_high": 10,
}

eq_IIR = AudioEqualizer(eq_params, mode="IIR")
eq_IIR.apply_equalization("music/thunderstruck.mp3")
eq_IIR.apply_equalization("music/from_the_start.wav")
eq_IIR.apply_equalization("music/hotel_california.mp3")

eq_FIR = AudioEqualizer(eq_params, mode="FIR")
eq_FIR.apply_equalization("music/thunderstruck.mp3")
eq_FIR.apply_equalization("music/from_the_start.wav")
eq_FIR.apply_equalization("music/hotel_california.mp3")
