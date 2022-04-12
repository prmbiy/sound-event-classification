# logmel spectrogram values
from random import sample


sample_rate = 16000
n_fft=(2560*44100)//sample_rate
hop_length=(694*44100)//sample_rate
n_mels = 128
fmin = 20
remove_codec_from_filename = True
fmax = sample_rate/2
num_cores = -1 # -1 implies take all available cores.