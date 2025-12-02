from typing import Tuple, List, Literal, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """
    def __init__(
            self, 
            partition: str, 
            config: dict, 
            tokenizer: H4Tokenizer
    ):
        self.config    = config
        self.partition = partition
        self.tokenizer = tokenizer

        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # --- PATH FIX: Check for 'text' subfolder ---
        root_path = os.path.join(config['root'], partition)
        possible_text = os.path.join(root_path, 'text')
        possible_transcript = os.path.join(root_path, 'transcript')
        
        if os.path.exists(possible_text) and len(os.listdir(possible_text)) > 0:
            self.text_dir = possible_text
        elif os.path.exists(possible_transcript) and len(os.listdir(possible_transcript)) > 0:
            self.text_dir = possible_transcript
        else:
            self.text_dir = root_path

        # If directory doesn't exist (e.g. test-clean for LM), handle gracefully
        if not os.path.exists(self.text_dir):
            print(f"Warning: {self.text_dir} does not exist. Dataset length will be 0.")
            self.text_files = []
        else:
            self.text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith('.npy')])

        subset_size = config.get('subset_size', len(self.text_files))
        if isinstance(subset_size, float): subset_size = int(subset_size * len(self.text_files))
        self.text_files = self.text_files[:subset_size]

        self.transcripts_shifted = []
        self.transcripts_golden  = []
        
        self.total_chars  = 0
        self.total_tokens = 0
        self.text_max_len = 0
        
        if len(self.text_files) > 0:
            print(f"Loading transcripts for {partition} partition from {self.text_dir}...")
            for file in tqdm(self.text_files):
                file_path = os.path.join(self.text_dir, file)
                transcript_arr = np.load(file_path, allow_pickle=True)
                transcript = "".join(transcript_arr.tolist())
                self.total_chars += len(transcript)
                tokenized = tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized)+1)
                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        self.length = len(self.transcripts_shifted)
        
    def get_avg_chars_per_token(self) -> float:
        return self.avg_chars_per_token
    
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden  = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden
    
    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shifted_transcripts, golden_transcripts = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in shifted_transcripts])
        padded_shifted = pad_sequence(shifted_transcripts, batch_first=True, padding_value=self.pad_token)
        padded_golden  = pad_sequence(golden_transcripts, batch_first=True, padding_value=self.pad_token)
        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(seed)
        prompts = []
        originals = []
        attempts = 0
        max_attempts = num_samples * 10 
        while len(prompts) < num_samples and attempts < max_attempts and len(self) > 0:
            idx = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:] 
            if len(tokens) < prompt_length:
                attempts += 1
                continue
            prompt_tokens = tokens[:prompt_length]
            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))
            attempts += 1
        if seed is not None:
            np.random.set_state(np_state)
        if not prompts: return torch.empty(0), []
        return torch.stack(prompts), originals


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

        # --- PATH FIX: Check for 'fbank' subfolder ---
        root_path = os.path.join(config['root'], partition)
        possible_fbank = os.path.join(root_path, 'fbank')
        
        if os.path.exists(possible_fbank) and len(os.listdir(possible_fbank)) > 0:
            self.fbank_dir = possible_fbank
        else:
            self.fbank_dir = root_path

        print(f"[{partition}] looking for features in: {self.fbank_dir}")
        self.fbank_files = sorted([f for f in os.listdir(self.fbank_dir) if f.endswith('.npy')])
        
        subset_size = config.get('subset_size', 1.0)
        if isinstance(subset_size, float): 
            subset_size = int(subset_size * len(self.fbank_files))
        else:
            subset_size = min(subset_size, len(self.fbank_files))
            
        self.fbank_files = self.fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        # Transcripts are only loaded for non-test partitions
        if self.partition != "test-clean":
            possible_text = os.path.join(root_path, 'text')
            possible_transcript = os.path.join(root_path, 'transcript')
            
            if os.path.exists(possible_text): self.text_dir = possible_text
            elif os.path.exists(possible_transcript): self.text_dir = possible_transcript
            else: self.text_dir = root_path

            self.text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith('.npy')])
            self.text_files = self.text_files[:subset_size]
            
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError(f"Mismatch: {len(self.fbank_files)} features vs {len(self.text_files)} transcripts")

        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []
        self.total_chars, self.total_tokens = 0, 0
        self.feat_max_len, self.text_max_len = 0, 0
        
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition: raise ValueError("global_stats needed for validation")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat = np.load(feat_path)
            feat = feat[:self.config['num_feats']] # Ensure correct dim
            self.feats.append(torch.FloatTensor(feat))
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)
                count += feat_tensor.shape[1]
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
        
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2/(count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        if self.config.get('specaug', False):
            self.time_mask = tat.TimeMasking(time_mask_param=config['specaug_conf']['time_mask_width_range'], iid_masks=True)
            self.freq_mask = tat.FrequencyMasking(freq_mask_param=config['specaug_conf']['freq_mask_width_range'], iid_masks=True)

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        feat = self.feats[idx]

        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = self.transcripts_shifted[idx]
            golden_transcript  = self.transcripts_golden[idx]

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch):
        batch_feats_raw, batch_shifted_raw, batch_golden_raw = zip(*batch)
        batch_feats = [feat.transpose(0, 1) for feat in batch_feats_raw]
        feat_lengths = torch.tensor([feat.shape[0] for feat in batch_feats])
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=self.pad_token)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean" and batch_shifted_raw[0] is not None:
            batch_shifted = list(batch_shifted_raw)
            batch_golden  = list(batch_golden_raw)
            transcript_lengths = torch.tensor([len(t) for t in batch_shifted])
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden  = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        if self.config.get("specaug", False) and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]): padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]): padded_feats = self.time_mask(padded_feats)
            padded_feats = padded_feats.permute(0, 2, 1)

        if self.partition == "test-clean":
            # For test set, return fewer items to match ASRTrainer.recognize unpacking
            return padded_feats, None, None, feat_lengths, None
            
        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
