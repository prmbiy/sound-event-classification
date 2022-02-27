import time
import wave
import librosa
import torch
import pyaudio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import Task5Model, configureTorchDevice, getSampleRateString
from augmentation.SpecTransforms import ResizeSpectrogram
from config import target_names, sample_rate, num_frames, gpu, threshold, num_classes, channels, n_fft, hop_length, n_mels, fmin, fmax, audio_segment_length, length_full_recording, resize, feature_type, permutation

labels = target_names
final_outputs = [False] * len(labels)
temp_filename = "test.wav"


def record(args):
    num_frames = args.num_frames
    feature_type = args.feature_type
    threshold = args.threshold
    mode = args.mode
    perm = args.permutation
    resizeSpec = ResizeSpectrogram(frames=num_frames)
    channel_means = np.load('./data/statistics/{}/channel_means_{}_{}.npy'.format(getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]))).reshape(1, -1, 1)
    channel_stds=np.load('./data/statistics/{}/channel_stds_{}_{}.npy'.format(getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]))).reshape(1, -1, 1)

    if args.gpu:
        device=configureTorchDevice()
    else:
        device=configureTorchDevice(False)

    model=Task5Model(num_classes).to(device)
    model.load_state_dict(torch.load(
        './model/{}k/model_{}_{}'.format(sample_rate/1000, feature_type, str(perm[0])+str(perm[1])+str(perm[2])), map_location=device))
    model=model.eval()

    length_full_recording=args.length_full_recording

    if mode == 'record':
        audio_segment_length=args.audio_segment_length
        chunk=sample_rate  # Process/Record in chunks of 44100 samples, ie 1 second at a time
        fs=sample_rate  # Process/Record at 44100 samples per second
        sample_format=pyaudio.paInt16  # 16 bits per sample
        p=pyaudio.PyAudio()
        sample_width=p.get_sample_size(sample_format)
        channels=args.channels
        print('Recording...')
        stream=p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

    elif mode == 'localfile':
        filename=args.filename
        print(f"Processing '{filename}'...")
        audio_segment_length=args.audio_segment_length

        wf=wave.open(filename, "rb")
        fs=wf.getframerate()
        channels=wf.getnchannels()
        sample_width=wf.getsampwidth()
        chunk=fs
        sr=fs
        # print(f'Sample rate for localfile: ', fs)

    frames=[]
    # frames = [b'110001001', b'10100010110', b'1001111011']
    length_recorded=0
    all_outputs = []
    while length_recorded < length_full_recording:
        if mode == 'record':
            data=stream.read(chunk)
        elif mode == 'localfile':
            data=wf.readframes(chunk)

        length_recorded += 1
        if len(frames) == audio_segment_length:
            frames=frames[1:] + [data]
        else:
            frames.append((data))
        if len(frames) < audio_segment_length:
            continue
        audio_segment=bytearray(frames[0])
        for i in range(audio_segment_length-1):
            audio_segment=audio_segment + bytearray(frames[i+1])

        wf1=wave.open(temp_filename, 'wb')
        wf1.setnchannels(channels)
        wf1.setsampwidth(sample_width)
        wf1.setframerate(fs)
        wf1.writeframes(audio_segment)
        wf1.close()

        wav=librosa.load(temp_filename, sr=sample_rate)[0]
        melspec=librosa.feature.melspectrogram(
            wav,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax)

        sample=librosa.core.power_to_db(melspec)
        if args.resize:
            sample=resizeSpec(sample)
        sample=(sample-channel_means)/channel_stds
        sample=torch.Tensor(sample)
        if len(sample.shape) <= 3:
            sample=torch.unsqueeze(sample, 0)
        inputs=sample.to(device)
        with torch.set_grad_enabled(False):
            outputs=model(inputs)
            outputs=torch.sigmoid(outputs)[0].detach().cpu().numpy()
            all_outputs.append(outputs)
            # print(length_recorded, outputs, [val if outputs[i]>threshold else '' for i, val in enumerate(labels)])
            print(f'{length_recorded}: ', end='')
            for i, val in enumerate(outputs):
                final_outputs[i]=final_outputs[i] or val > threshold
                if(val>threshold):
                    print(f'{labels[i]}, ', end='')
            print(f' ')
        # if mode=='localfile':
        #     time.sleep(1)

    if mode == 'record':
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()
        print('Finished recording')
    elif mode == 'localfile':
        wf.close()
        print(f'Finished processing {length_full_recording}s of {filename} ')
    final_prediction=[labels[i] if val else "" ]
    print('final_predictions: ', end='')
    for i, val in enumerate(final_outputs):
        if val:
            print(f'{labels[i]}, ', end='')
    print()
    heatmap_outputs = np.array(all_outputs).T
    ax = plt.axes()
    plt.imshow(heatmap_outputs)
    plt.colorbar()
    ax.set_xticks(np.arange(length_full_recording - audio_segment_length + 1), labels=[f'{i}' for i in range(audio_segment_length, length_full_recording+1, 1)])
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    if mode == 'localfile':
        plt.title(filename)
        plt.savefig(filename.replace('.wav', '.png'))
    else:
        plt.title('Recording outputs')
        plt.show()
        plt.savefig('output.png')

if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description='For making realtime predictons.')
    subparsers=parser.add_subparsers(
        dest='mode')


    # mode RECORD
    parser_record=subparsers.add_parser('record')
    parser_record.add_argument(
        '-f', '--feature_type', type=str, default=feature_type)
    parser_record.add_argument(
        '-n', '--num_frames', type=int, default=num_frames)
    parser_record.add_argument(
        '-t', '--threshold', type=float, default=threshold)
    parser_record.add_argument('-g', '--gpu', type=bool, default=gpu)
    parser_record.add_argument(
        '-l', '--length_full_recording', type=int, default=length_full_recording)
    parser_record.add_argument(
        '-a', '--audio_segment_length', type=int, default=audio_segment_length)
    parser_record.add_argument('-r', '--resize', type=bool, default=resize)
    parser_record.add_argument('-c', '--channels', type=int, default=channels)


    # mode LOCALFILE
    parser_localfile=subparsers.add_parser('localfile')
    parser_localfile.add_argument(
        '-f', '--feature_type', type=str, default=feature_type)
    parser_localfile.add_argument('-fi', '--filename', type=str,
                                  help="Specify path of input audio file")
    parser_localfile.add_argument(
        '-n', '--num_frames', type=int, default=num_frames)
    parser_localfile.add_argument(
        '-t', '--threshold', type=float, default=threshold)
    parser_localfile.add_argument('-g', '--gpu', type=bool, default=gpu)
    parser_localfile.add_argument(
        '-a', '--audio_segment_length', type=int, default=audio_segment_length)
    parser_localfile.add_argument('-r', '--resize', type=bool, default=resize)
    parser_localfile.add_argument(
        '-l', '--length_full_recording', type=int, default=length_full_recording)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)

    args=parser.parse_args()

    record(args)
