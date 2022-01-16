import wave
import librosa
import torch
import pyaudio
import numpy as np
from utils import Task5Model
from augmentation.SpecTransforms import ResizeSpectrogram

num_frames = 636
feature_type = "logmelspec"
threshold = 0.1
resize = ResizeSpectrogram(frames=num_frames)
channel_means = np.load('./data/statistics/channel_means_{}.npy'.format(feature_type)).reshape(1,-1,1)
channel_stds = np.load('./data/statistics/channel_stds_{}.npy'.format(feature_type)).reshape(1,-1,1)
labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact',
            '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']
final_outputs = [False] * len(labels)

cuda = False
device = torch.device('cuda:0' if cuda else 'cpu')
print('Device: ', device)
model = Task5Model(8).to(device)
model.load_state_dict(torch.load('./models/model_{}'.format(feature_type), map_location = device))
model = model.eval()


# filename = "/notebooks/sonyc_urban_sound_tagging/validate/00_000066.wav"
# print(f"Processing '{filename}'...")
# wf = wave.open(filename, "rb")
# fs = wf.getframerate()
# channels = wf.getnchannels()
# seconds = 3
# chunk = seconds * fs
# sr = 44100

chunk = 44100  # Record in chunks of 44100 samples, ie 1 second at a time
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "test.wav"
full_recording_length = 10
audio_segment_length = 3
length_recorded = 0

p = pyaudio.PyAudio()
sample_width = p.get_sample_size(sample_format)

frames = []
print('Recording...')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

while length_recorded < full_recording_length:
    data = stream.read(chunk)
    length_recorded+=1
    if len(frames) < audio_segment_length:
        frames.append((data))
        continue
    frames = frames[1:] + [data]
    audio_segment = bytearray(frames[0])
    for i in range(audio_segment_length-1):
        audio_segment = audio_segment + bytearray(frames[i+1])
    print(np.frombuffer(audio_segment).shape)
    wf1 = wave.open(f'test.wav', 'wb')
    wf1.setnchannels(channels)
    wf1.setsampwidth(sample_width)
    wf1.setframerate(fs)
    wf1.writeframes(audio_segment)
    wf1.close()
    wav = librosa.load(f'test.wav', sr=44100)[0]
    melspec = librosa.feature.melspectrogram(
            wav,
            sr=fs,
            n_fft=2560,
            hop_length=694,
            n_mels=128,
            fmin=20,
            fmax=22050)
    sample = librosa.core.power_to_db(melspec)
    sample = resize(sample)
    sample = (sample-channel_means)/channel_stds
    sample = torch.Tensor(sample)
    print(sample.shape)
    if len(sample.shape)<=3:
        sample = torch.unsqueeze(sample, 0)
    inputs = sample.to(device)
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
        print(length_recorded, outputs)
        for i, val in enumerate(outputs):
            final_outputs[i] = final_outputs[i] or val>threshold
# ctr = 1
# data = wf.readframes(chunk)
# while data:
#     wf1 = wave.open(f'test.wav', 'wb')
#     wf1.setnchannels(channels)
#     wf1.setsampwidth(wf.getsampwidth())
#     wf1.setframerate(fs)
#     wf1.writeframes(data)
#     wf1.close()
#     wav = librosa.load(f'test.wav', sr=44100)[0]
#     melspec = librosa.feature.melspectrogram(
#             wav,
#             sr=sr,
#             n_fft=2560,
#             hop_length=694,
#             n_mels=128,
#             fmin=20,
#             fmax=22050)
    # sample = librosa.core.power_to_db(melspec)    
    # sample = (sample-channel_means)/channel_stds
    # sample = torch.Tensor(sample)
    # print(sample.shape)
    # if len(sample.shape)<=3:
    #     sample = torch.unsqueeze(sample, 0)
    # inputs = sample.to(device)
    # with torch.set_grad_enabled(False):
    #     outputs = model(inputs)
    #     outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
    #     print(ctr, outputs)
    # data = wf.readframes(chunk)
    # ctr+=1


# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')
final_prediction = [labels[i] if val else "" for i, val in enumerate(final_outputs)]
print('final_prediction', final_prediction)