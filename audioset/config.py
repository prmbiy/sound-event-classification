import numpy as np
batch_size = 16
num_workers = 2
sample_rate = 88200
feature_type = 'logmelspec'
num_bins = 128
resize = True
learning_rate = 0.001
amsgrad = True
verbose = True
patience = 5
epochs = 50
early_stopping = 10
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
# num_frames = 200
num_frames = int(np.ceil(sample_rate*length_full_recording/hop_length))

permutation = [0, 1, 2, 3, 4]
workspace = '/notebooks/sound-event-classification/audioset'
target_names = ['breaking', 'chatter', 'crying_sobbing', 'emergency_vehicle', 'explosion', 'gunshot_gunfire', 'motor_vehicle_road', 'screaming', 'siren', 'others']
num_classes = len(target_names)
