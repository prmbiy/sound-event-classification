import numpy as np
num_workers = 2
feature_type = 'logmelspec'
num_bins = 128
resize = True
learning_rate = 0.001
amsgrad = True
verbose = True
patience = 5
epochs = 50
early_stopping = 10
gpu = False
channels = 2
length_full_recording = 10
audio_segment_length = 3
seed = 42

sample_rate = 16000
threshold = 0.9
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
batch_size = num_classes * 1 #for balancedbatchsampler, for every batch to have equal number of samples, the size of each batch should be a multiple of the num of classes
grad_acc_steps = 2

voting = 'simple_average'
# voting = 'weighted_average'
weights = [2, 3, 5]
sum_weights = sum(weights)
normalised_weights = np.array(weights)/sum_weights

# paperspace
pann_encoder_ckpt_path = '/notebooks/sound-event-classification/audioset/model/Cnn10_mAP=0.380.pth'
model_arch = 'pann_cnn10'
resume_training = 'yes'