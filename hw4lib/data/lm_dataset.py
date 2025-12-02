from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
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

        # Handle non-existent directories gracefully
        if not os.path.exists(self.text_dir):
            print(f"Warning: {self.text_dir} does not exist. Dataset length will be 0.")
            self.text_files = []
        else:
            self.text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith('.npy')])

        # FIX: Check 'subset' key (matches yaml) OR 'subset_size'
        subset_val = config.get('subset', config.get('subset_size', 1.0))
        if isinstance(subset_val, float): 
            subset_len = int(subset_val * len(self.text_files))
        else:
            subset_len = min(subset_val, len(self.text_files))
            
        self.text_files = self.text_files[:subset_len]

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
                
                # Store as lists first for memory efficiency during loading
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
        
        # Guard against infinite loops if dataset is empty or prompts too short
        attempts = 0
        max_attempts = num_samples * 20 
        
        while len(prompts) < num_samples and attempts < max_attempts and len(self) > 0:
            idx = np.random.randint(0, len(self))
            # Get original token list (remove SOS from shifted)
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
            
        if not prompts: 
            return torch.empty(0), []
            
        return torch.stack(prompts), originals
