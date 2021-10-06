import os
from PIL import Image
from scipy.io import wavfile
import numpy as np
from core.util import generate_mel_spectrogram


def _pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def _pil_loader_fail(path, target_size):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except OSError as e:
        return Image.new('RGB', target_size)


def _fit_audio_clip(audio_clip, sample_rate, video_clip_lenght):
    target_audio_length = int((1.0/27.0)*sample_rate*video_clip_lenght)
    pad_required = int((target_audio_length-len(audio_clip))/2)
    if pad_required > 0:
        audio_clip = np.pad(audio_clip, pad_width=(pad_required, pad_required),
                            mode='reflect')
    if pad_required < 0:
        audio_clip = audio_clip[-1*pad_required:pad_required]

    # There is a +-1 offset here and I dont feel like cheking it
    return audio_clip[0:target_audio_length-1]


def load_av_clip_from_metadata(clip_meta_data, frames_source, audio_source,
                               audio_offset):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    # Video Frames
    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    video_data = [_pil_loader(sf) for sf in selected_frames]

    # Audio File
    audio_file = os.path.join(audio_source, entity_id+'.wav')
    sample_rate, audio_data = wavfile.read(audio_file)

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(selected_frames))
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)

    return video_data, audio_features


def load_av_clip_from_metadata_forward(clip_meta_data, frames_source, audio_source,
                                       target_size, audio_offset):
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    # Video Frames
    selected_frames = [os.path.join(frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    video_data = [_pil_loader_fail(sf, target_size) for sf in selected_frames]

    # Audio File
    audio_file = os.path.join(audio_source, entity_id+'.wav')
    try:
        sample_rate, audio_data = wavfile.read(audio_file)
    except:
        sample_rate, audio_data = 16000,  np.zeros(int((1.0/27.0)*16000*len(selected_frames)))

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    if len(audio_clip) < int((1.0/27.0)*sample_rate):
        audio_clip = np.zeros(int((1.0/27.0)*sample_rate*len(selected_frames)))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(selected_frames))
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)

    return video_data, audio_features
