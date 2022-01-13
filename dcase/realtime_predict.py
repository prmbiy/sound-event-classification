import wave
import librosa
import torch
import numpy as np
from utils import Task5Model
from augmentation.SpecTransforms import ResizeSpectrogram

num_frames = 636
feature_type = "logmelspec"
reize = ResizeSpectrogram(frames=num_frames)
channel_means = np.load('./data/statistics/channel_means_{}.npy'.format(feature_type)).reshape(1,-1,1)
channel_stds = np.load('./data/statistics/channel_stds_{}.npy'.format(feature_type)).reshape(1,-1,1)



cuda = True
device = torch.device('cuda:0' if cuda else 'cpu')
print('Device: ', device)
model = Task5Model(8).to(device)
model.load_state_dict(torch.load('./models/model_{}'.format(feature_type)))
model = model.eval()


filename = "/notebooks/sonyc_urban_sound_tagging/validate/00_000066.wav"
print(f"Processing '{filename}'...")
wf = wave.open(filename, "rb")
fs = wf.getframerate()
channels = wf.getnchannels()
seconds = 3
chunk = seconds * fs
sr = 44100

ctr = 1
data = wf.readframes(chunk)
while data:
    wf1 = wave.open(f'test.wav', 'wb')
    wf1.setnchannels(channels)
    wf1.setsampwidth(wf.getsampwidth())
    wf1.setframerate(fs)
    wf1.writeframes(data)
    wf1.close()
    wav = librosa.load(f'test.wav', sr=44100)[0]
    melspec = librosa.feature.melspectrogram(
            wav,
            sr=sr,
            n_fft=2560,
            hop_length=694,
            n_mels=128,
            fmin=20,
            fmax=22050)
    sample = librosa.core.power_to_db(melspec)    
    sample = (sample-channel_means)/channel_stds
    sample = torch.Tensor(sample)
    print(sample.shape)
    if len(sample.shape)<=3:
        sample = torch.unsqueeze(sample, 0)
    inputs = sample.to(device)
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
        print(ctr, outputs)
    data = wf.readframes(chunk)
    ctr+=1


print('done')

# stream.close()
# p.terminate()