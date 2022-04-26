import numpy as np

__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

num_workers = 2
feature_type = 'logmelspec'
num_bins = 128
resize = True
learning_rate = 0.001
amsgrad = True
verbose = True
patience = 5
epochs = 50
early_stopping = 20
gpu = False
channels = 2
length_full_recording = 10
audio_segment_length = 3
seed = 42

#                           441,000        160,000
# nfft/window_len           2560        7056
# hop_len                   694         1912
# num_frames                656         84
sample_rate = 16000
threshold = 0.9
n_fft = (2560*sample_rate)//44100
hop_length = (694*sample_rate)//44100
n_mels = 128
fmin = 20
fmax = 22050
# num_frames = 200
num_frames = int(np.ceil(sample_rate*length_full_recording/hop_length))

permutation = [0, 1, 2, 3, 4]
workspace = '/notebooks/sound-event-classification/audioset'
target_names = ['breaking', 'chatter', 'crying_sobbing', 'emergency_vehicle',
                'explosion', 'gunshot_gunfire', 'motor_vehicle_road', 'screaming', 'siren', 'others']
num_classes = len(target_names)
# for balancedbatchsampler, for every batch to have equal number of samples, the size of each batch should be a multiple of the num of classes
batch_size = 32
grad_acc_steps = 1

# voting = 'simple_average'
voting = 'weighted_average'
weights = [2, 3, 5]
sum_weights = sum(weights)
normalised_weights = np.array(weights)/sum_weights

# CBAM
use_cbam = True
cbam_channels = 512
cbam_reduction_factor = 16
cbam_kernel_size = 7

# MEDIAN FILTERING
use_median_filter = False

# paperspace
pann_cnn10_encoder_ckpt_path = '/notebooks/sound-event-classification/audioset/model/Cnn10_mAP=0.380.pth'
pann_cnn14_encoder_ckpt_path = '/notebooks/sound-event-classification/audioset/model/Cnn14_mAP=0.431.pth'
model_arch = 'mobilenetv2'
resume_training = 'yes'
