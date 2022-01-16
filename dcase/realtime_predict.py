import wave
import librosa
import torch
import pyaudio
import argparse
import numpy as np
from utils import Task5Model
from augmentation.SpecTransforms import ResizeSpectrogram
labels = ['1_engine', '2_machinery-impact', '3_non-machinery-impact',
          '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog']
final_outputs = [False] * len(labels)
temp_filename = "test.wav"


def record(args):
    num_frames = args.num_frames
    feature_type = args.feature_type
    threshold = args.threshold
    resize = ResizeSpectrogram(frames=num_frames)
    channel_means = np.load(
        './data/statistics/channel_means_{}.npy'.format(feature_type)).reshape(1, -1, 1)
    channel_stds = np.load(
        './data/statistics/channel_stds_{}.npy'.format(feature_type)).reshape(1, -1, 1)

    cuda = args.gpu
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    model = Task5Model(8).to(device)
    model.load_state_dict(torch.load(
        './models/model_{}'.format(feature_type), map_location=device))
    model = model.eval()

    chunk = 44100  # Record in chunks of 44100 samples, ie 1 second at a time
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    full_recording_length = args.length_full_recording
    audio_segment_length = args.audio_segment_length
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
        length_recorded += 1
        if len(frames) < audio_segment_length:
            frames.append((data))
            continue
        frames = frames[1:] + [data]
        audio_segment = bytearray(frames[0])
        for i in range(audio_segment_length-1):
            audio_segment = audio_segment + bytearray(frames[i+1])

        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(fs)
        wf.writeframes(audio_segment)
        wf.close()

        wav = librosa.load(temp_filename, sr=44100)[0]
        melspec = librosa.feature.melspectrogram(
            wav,
            sr=fs,
            n_fft=2560,
            hop_length=694,
            n_mels=128,
            fmin=20,
            fmax=22050)

        sample = librosa.core.power_to_db(melspec)
        if args.resize:
            sample = resize(sample)
        sample = (sample-channel_means)/channel_stds
        sample = torch.Tensor(sample)
        if len(sample.shape) <= 3:
            sample = torch.unsqueeze(sample, 0)
        inputs = sample.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
            print(length_recorded, outputs)
            for i, val in enumerate(outputs):
                final_outputs[i] = final_outputs[i] or val > threshold

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()
    print('Finished recording')
    final_prediction = [labels[i] if val else "" for i,
                        val in enumerate(final_outputs)]
    print('final_prediction', final_prediction)


def localfile(args):
    num_frames = args.num_frames
    feature_type = args.feature_type
    threshold = args.threshold
    resize = ResizeSpectrogram(frames=num_frames)
    channel_means = np.load(
        './data/statistics/channel_means_{}.npy'.format(feature_type)).reshape(1, -1, 1)
    channel_stds = np.load(
        './data/statistics/channel_stds_{}.npy'.format(feature_type)).reshape(1, -1, 1)

    cuda = args.gpu
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    model = Task5Model(8).to(device)
    model.load_state_dict(torch.load(
        './models/model_{}'.format(feature_type), map_location=device))
    model = model.eval()


    filename = args.filename
    print(f"Processing '{filename}'...")
    wf = wave.open(filename, "rb")
    fs = wf.getframerate()
    channels = wf.getnchannels()
    audio_segment_length = args.audio_segment_length
    chunk = 44100
    sr = 44100
    length_recorded = 0
    full_recording_length = 10
    frames = []

    while length_recorded < full_recording_length:
        data = wf.readframes(chunk)
        length_recorded += 1
        if len(frames) < audio_segment_length:
            frames.append((data))
            continue
        frames = frames[1:] + [data]
        audio_segment = bytearray(frames[0])
        for i in range(audio_segment_length-1):
            audio_segment = audio_segment + bytearray(frames[i+1])
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(wf.getsampwidth())
        wf.setframerate(fs)
        wf.writeframes(data)
        wf.close()
        wav = librosa.load(temp_filename, sr=44100)[0]
        melspec = librosa.feature.melspectrogram(
                wav,
                sr=sr,
                n_fft=2560,
                hop_length=694,
                n_mels=128,
                fmin=20,
                fmax=22050)
        sample = librosa.core.power_to_db(melspec)
        if args.resize:
            sample = resize(sample)
        sample = (sample-channel_means)/channel_stds
        sample = torch.Tensor(sample)
        if len(sample.shape)<=3:
            sample = torch.unsqueeze(sample, 0)
        inputs = sample.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)[0].detach().cpu().numpy()
            for i, val in enumerate(outputs):
                final_outputs[i] = final_outputs[i] or val > threshold

    final_prediction = [labels[i] if val else "" for i,
                        val in enumerate(final_outputs)]
    print('final_prediction', final_prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='For making realtime predictons.')
    subparsers = parser.add_subparsers(
        dest='mode')


    # mode RECORD
    parser_record = subparsers.add_parser('record')
    parser_record.add_argument(
        '-f', '--feature_type', type=str, default='logmelspec')
    parser_record.add_argument('-n', '--num_frames', type=int, default=636)
    parser_record.add_argument('-t', '--threshold', type=float, default=0.1)
    parser_record.add_argument('-g', '--gpu', type=bool, default=False)
    parser_record.add_argument(
        '-l', '--length_full_recording', type=int, default=10)
    parser_record.add_argument(
        '-a', '--audio_segment_length', type=int, default=3)
    parser_record.add_argument('-r', '--resize', type=bool, default=True)


    # mode LOCALFILE
    parser_localfile = subparsers.add_parser('localfile')
    parser_localfile.add_argument(
        '-f', '--feature_type', type=str, default='logmelspec')
    parser_localfile.add_argument('-fi', '--filename', type=str,
                                  help="Specify path of input audio file")
    parser_localfile.add_argument('-n', '--num_frames', type=int, default=636)
    parser_localfile.add_argument('-t', '--threshold', type=float, default=0.1)
    parser_localfile.add_argument('-g', '--gpu', type=bool, default=False)
    parser_localfile.add_argument(
        '-a', '--audio_segment_length', type=int, default=3)
    parser_localfile.add_argument('-r', '--resize', type=bool, default=True)

    
    args = parser.parse_args()

    if args.mode == "record":
        record(args)
    elif args.mode == "localfile":
        localfile(args)
