import numpy as np
batch_size = 16
num_workers = 2
labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact',
          '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']
sample_rate = 16000
feature_type = 'logmelspec'
num_bins = 128
resize = True
num_classes = 10
learning_rate = 0.001
amsgrad = True
verbose = True
patience = 5
epochs = 100
threshold = 0.3
gpu = False
channels = 2
length_full_recording = 10
audio_segment_length = 9
seed = 42

n_fft=2560
hop_length=694
n_mels=128
fmin=20
fmax=sample_rate/2
num_frames = int(np.ceil(sample_rate/hop_length))

permutation = [0, 1, 2, 3, 4]