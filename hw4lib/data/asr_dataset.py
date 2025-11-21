from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Similar to LMDataset transcript handling
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H4Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''

class ASRDataset(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        self.config    = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        self.fbank_dir = os.path.join(config['root'], partition, 'fbank')
        
        self.fbank_files = sorted([f for f in os.listdir(self.fbank_dir) if f.endswith('.npy')])
        
        subset_size = config.get('subset_size', len(self.fbank_files))
        self.fbank_files = self.fbank_files[:subset_size]
        
        self.length = len(self.fbank_files)

        if self.partition != "test-clean":
            self.text_dir = os.path.join(config['root'], partition, 'text')

            self.text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith('.npy')])
            
            self.text_files = self.text_files[:subset_size]
            
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []
        
        self.total_chars  = 0
        self.total_tokens = 0
        
        self.feat_max_len = 0
        self.text_max_len = 0
        
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat = np.load(feat_path)

            feat = feat[:self.config['num_feats']]

            self.feats.append(torch.FloatTensor(feat))

            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)
                batch_count = feat_tensor.shape[1]
                count += batch_count
                
                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            if self.partition != "test-clean":
                text_path = os.path.join(self.text_dir, self.text_files[i])
                transcript_arr = np.load(text_path, allow_pickle=True)
                transcript = "".join(transcript_arr.tolist())

                self.total_chars += len(transcript)

                tokenized = tokenizer.encode(transcript)

                self.total_tokens += len(tokenized)

                self.text_max_len = max(self.text_max_len, len(tokenized)+1)
                
                self.transcripts_shifted.append(torch.LongTensor([self.sos_token] + tokenized))
                self.transcripts_golden.append(torch.LongTensor(tokenized + [self.eos_token]))

        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        
        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2/(count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.feats[idx]

        if self.config['norm'] == 'global_mvn':
            assert self.global_mean is not None and self.global_std is not None, "Global mean and std must be computed before normalization"
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass
        
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = self.transcripts_shifted[idx]
            golden_transcript  = self.transcripts_golden[idx]

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_feats_raw, batch_shifted_raw, batch_golden_raw = zip(*batch)

        batch_feats = [feat.transpose(0, 1) for feat in batch_feats_raw]

        feat_lengths = torch.tensor([feat.shape[0] for feat in batch_feats])

        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=self.pad_token)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = list(batch_shifted_raw)
            batch_golden  = list(batch_golden_raw)

            transcript_lengths = torch.tensor([len(t) for t in batch_shifted])

            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden  = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)

            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            padded_feats = padded_feats.permute(0, 2, 1)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths