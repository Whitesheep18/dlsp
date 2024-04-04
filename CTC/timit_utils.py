import torch
from torch.nn.utils.rnn import pad_sequence
import os
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import torch.nn as nn

symbol_to_phoneme = {
    "aa": "aa", "ao": "aa", "ax": "ah", "ax-h": "ah", "axr": "er", "hv": "hh", "ix": "ih", "ux": "uw",
    "bcl": "h#", "dcl": "h#", "gcl": "h#", "h#": "h#", "pcl": "h#", "tcl": "h#", "kcl": "h#", "q": "h#", "pau": "h#", "epi": "h#",
    "em": "m", "en": "n", "eng": "ng", "nx": "n", "el": "l", "l": "l", "b": "b", "ch": "ch", "d": "d", 
    "dh": "dh", "dx": "dx", "eh": "eh", "ey": "ey", "f": "f", "g": "g", "hh": "hh", "ih": "ih", "iy": "iy", "jh": "jh", 
    "k": "k", "m": "m", "n": "n", "ng": "ng", "ow": "ow", "oy": "oy", "p": "p", "r": "r", "s": "s", "sh": "sh", "t": "t", 
    "th": "th", "uh": "uh", "uw": "uw", "v": "v", "w": "w", "y": "y", "z": "z", "zh": "zh",
    'ay': 'ay', 'ae':'ae', 'er':'er', 'aw':'aw', "ah":"ah"
}

def extract_frame(wav, hd, tl):
    return wav[hd:tl]

def get_frames(wav, phonemes):
    frames = [extract_frame(wav.squeeze(0), hd, tl) for hd, tl, _ in phonemes]
    padded_frames = pad_sequence(frames, batch_first=True).unsqueeze(-1)
    return padded_frames


def load_timit(file: str):
    data_path = os.path.splitext(file)[0]
    with open(data_path + '.txt', 'r') as txt_file:
        _, __, transcript = next(iter(txt_file)).strip().split(" ", 2)
    with open(data_path + '.wrd', 'r') as word_file:
        words = [l.strip().split(' ') for l in word_file]
        words = [(int(hd), int(tl), w) for (hd, tl, w) in words]
    with open(data_path + '.phn', 'r') as phn_file:
        phonemes = [l.strip().split(' ') for l in phn_file]
        phonemes = [(int(hd), int(tl), w) for (hd, tl, w) in phonemes]
    wav, sr = torchaudio.load(data_path + '.wav')
    return data_path, wav, transcript, words, phonemes


class Timit(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.walker = list(Path(root).rglob('*.wav'))
        self.phonemes = ['blank', 'ay', 'l', 'dcl', 'gcl', 'ch', 'm', 'v', 
                        'nx', 'kcl', 'ae', 'ih', 'b', 'd', 'el', 'k', 'n', 
                        'uh', 'hh', 'ux', 'y', 'f', 'en', 'tcl', 'pau', 'aa',
                        'ix', 'sh', 'eng', 'w', 's', 'ow', 'dh', 'ng', 'dx', 
                        'ax', 'ao', 'jh', 't', 'g', 'axr', 'h#', 'z', 'p',
                            'zh', 'th', 'ah', 'uw', 'em', 'bcl', 'ey', 'aw', 
                            'q', 'pcl', 'r', 'iy', 'eh', 'oy', 'hv', 'er', 
                            'ax-h', 'epi']
        self.phonemes_to_idx = {phoneme: idx for idx, phoneme in enumerate(self.phonemes)}
        self.idx_to_phonemes = {idx: phoneme for phoneme, idx in self.phonemes_to_idx.items()}

    def __getitem__(self, item):
        data_path, wav, transcript, _, phonemes =  load_timit(self.walker[item])
        return data_path, wav, transcript, get_frames(wav, phonemes), torch.tensor([self.phonemes_to_idx[p] for _, _, p in phonemes])

    def __len__(self):
        return len(self.walker)
    

def get_log_energy(wav, win_length=16, hop_length=8):
    avg = nn.AvgPool1d(win_length, stride=hop_length, ceil_mode=True)
    log_energy = torch.log(avg(wav**2)*win_length) # multiply by  len=16 to get sum instead of avg
    log_energy = torch.cat([log_energy, torch.zeros(1, 1)], dim=1)
    return log_energy # [1, T]

def get_derivatives(spectogram):
    diff = torch.diff(spectogram, dim=2)
    diff = torch.cat([diff, torch.zeros(1, diff.shape[1], 1)], dim=2)
    return diff

def preprocess(wav):
    """From wav to feature representation described in 5. Experiments / 5.1 Data"""
    spectogram = torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True, melkwargs={"win_length":16, "hop_length":8})(wav)
    spectogram = torch.concat([spectogram, get_log_energy(wav).unsqueeze(0)], dim=1)
    spectogram = torch.cat([spectogram, get_derivatives(spectogram)], dim=1)

    # normalize to have zero mean and unit variance
    spectogram = (spectogram - spectogram.mean()) / spectogram.std() #maybe mean and std is taken from the whole dataset?

    return spectogram