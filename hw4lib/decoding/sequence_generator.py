Wil
xoryax
Do Not Disturb

Wil — 10/28/25, 1:50 PM
Like 2.5
It’s on Wandb 
WeairdCow — 10/28/25, 1:51 PM
Bro mine is so ass
Wil — 10/28/25, 1:51 PM
Why is my training time 10 min
Huh
It's been like 4min
Image
WeairdCow — 10/29/25, 10:41 PM
Attachment file type: unknown
11_485_11_685_11_785_HW3P2_F25_Student_Starter_Notebook (2).ipynb
1.76 MB
Wil — 10/30/25, 10:59 AM
I retried running your code
it just ain't going down from 60
WeairdCow — 11/1/25, 11:03 PM
Whered u go
Wil — 11/1/25, 11:04 PM
I went downstairs to fill Brit’s
WeairdCow — 11/5/25, 9:22 PM
https://drive.google.com/file/d/1r489AbZK9pJ9p384YRgXNTxZqxIrXH6K/view?usp=sharing
Google Docs
11_485_11_685_11_785_HW3P2_F25_Student_Starter_Notebook.ipynb
Colab notebook
WeairdCow — 11/6/25, 10:56 PM
Image
Wil — 11/7/25, 4:29 PM
How did u do CTC forward and backwards? I keep getting wrong answer
WeairdCow — 11/7/25, 4:37 PM
Forwarded
import numpy as np

class GreedySearchDecoder(object):
    def __init__(self, symbol_set):
        self.symbol_set = symbol_set
Expand
CTCDecoding.py
3 KB
WeairdCow — 11/7/25, 5:23 PM
import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
Expand
CTC.py
15 KB
WeairdCow — 11/7/25, 5:33 PM
Image
Wil — 11/21/25, 2:22 PM
Did u hit early cutoff?
WeairdCow — 11/21/25, 2:25 PM
Yeah its 50% right
Just chdck leaderboard
Wil — 11/21/25, 2:25 PM
how long did u train fro?
WeairdCow — 11/21/25, 2:26 PM
Not that ling
Wil — 11/21/25, 2:31 PM
I'm at OH but TA just ain't here 😭
😭
Wil — 11/21/25, 6:45 PM
Could u send your transformer file?
I swear this just isn't working
WeairdCow — 11/21/25, 6:47 PM
import torch.nn as nn
import torch
import random
import math
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
Expand
message.txt
29 KB
WeairdCow — 11/23/25, 6:28 PM
our fridge keeps makingw eird noises
if we keep hearing them i might call maintenance
Wil — 11/23/25, 6:28 PM
Uhh that don’t sound good
Wil — 11/23/25, 8:31 PM
!git pull
import importlib
import hw4lib.model.transformers as tr
importlib.reload(tr)
from hw4lib.model.transformers import EncoderDecoderTransformer
WeairdCow — 11/24/25, 1:15 AM
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
... (286 lines left)
Collapse
message.txt
17 KB
Wil — 11/24/25, 5:29 PM
did u do problem 1 of the hw for 10701? My friend Gordon was wondering how it should be done
WeairdCow — 11/24/25, 5:48 PM
Which part
Its multi part
Wil — 11/24/25, 5:48 PM
entire thing lol
WeairdCow — 11/24/25, 5:59 PM
Forwarded
just check against all when done
Attachment file type: acrobat
10701_hw6.pdf
259.04 KB
Wil — 11/24/25, 6:09 PM
it doesn't reveal
WeairdCow — 11/24/25, 6:10 PM
?
Oh just wait for it to load
Wil — 11/24/25, 6:10 PM
oh alr
WeairdCow — 11/24/25, 8:11 PM
Try doing it on my proj
See if it works
Wil — 11/24/25, 8:12 PM
alr I'll try later
WeairdCow — 11/26/25, 5:00 PM
How much butter dyt we need
Amy has less than we thought
Wil — 11/26/25, 5:01 PM
Less than a stick
We could also just buy some at Scotty
WeairdCow — 11/30/25, 9:36 PM
Forwarded
Image
Wil — Yesterday at 11:01 AM
Could u send me ur basetrainer?
I think the issue with wandb might be with basetrainer
Wil — Yesterday at 6:43 PM
nvm it wasn't basetrainer since it's just default code
WeairdCow — Yesterday at 7:05 PM
oh gg i didnt see this mb
Wil — Yesterday at 7:06 PM
I think I got it work
holy I was digging through forums for it
WeairdCow — Yesterday at 7:08 PM
what was the problem
Wil — Yesterday at 7:08 PM
import wandb

# 1. Save the original wandb.init function so we don't lose it
_original_wandb_init = wandb.init

# 2. Define a new wrapper function that injects your entity
def forced_entity_wandb_init(*args, **kwargs):
    # Force the entity to be the one that works
    kwargs['entity'] = "alexzhen-cmu" 
    
    # Optional: Force the project name if needed (uncomment if necessary)
    # kwargs['project'] = "Wilson"
    
    print(f"Intercepted wandb.init! Forcing entity to: {kwargs['entity']}")
    return _original_wandb_init(*args, **kwargs)

# 3. Overwrite the wandb library's init function with our wrapper
wandb.init = forced_entity_wandb_init
 
I just overwrote the wandb init
WeairdCow — Yesterday at 7:09 PM
oh lmfao
Wil — Yesterday at 7:09 PM
It's legit their problem
no reason someone should have to completely overwrite their init
Did u get dinner yet?
I'm sick of chipotle after eating it for 4 meals straight
WeairdCow — Yesterday at 7:12 PM
nah but im in anime rn
LMFAO
bro just order block
Wil — Yesterday at 7:12 PM
yea imma prob do that
﻿
WeairdCow
weairdcow
stray catcher
 
 
 
ilmcl
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement greedy search
        x = x.to(self.device)

        scores = torch.zeros(x.size(0), device=x.device)
        finished = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        steps = self.max_length - x.size(1)

        for _ in range(steps):
            if finished.all():
                break

            logits = self.score_fn(x)
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)

            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)

            next_tokens = torch.argmax(log_probs, dim=-1)

            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = next_tokens == self.tokenizer.eos_id
            finished = finished | is_eos

        return x, scores # Remove once implemented # Remove once implemented

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement beam search
        x = x.to(self.device)
        batch_size, init_len = x.shape
        
        logits0 = self.score_fn(x)
        logits0 = self._apply_repeat_penalty(logits0, x, repeat_penalty)
        if temperature != 1.0:
            logits0 = logits0 / temperature
        log_probs0 = torch.log_softmax(logits0, dim=-1)
        vocab_size = log_probs0.size(-1)

        topk_log_probs, next_tokens = torch.topk(logits0, k=beam_width, dim=-1)

        sequences = x.unsqueeze(1).expand(batch_size, beam_width, init_len).clone()
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)
        scores = topk_log_probs.clone()
        finished = next_tokens.eq(self.tokenizer.eos_id)

        max_steps = self.max_length - sequences.size(-1)
        for _ in range(max_steps):
            if finished.all():
                break

            cur_len = sequences.size(-1)

            per_beam_log_probs = []
            for k in range(beam_width):
                logits_k = self.score_fn(sequences[:, k, :])
                logits_k = self._apply_repeat_penalty(logits_k, sequences[:, k, :], repeat_penalty)
                if temperature != 1.0:
                    logits_k = logits_k / temperature
                log_probs_k = torch.log_softmax(logits_k, dim=-1)
                per_beam_log_probs.append(log_probs_k)
            next_token_log_probs = torch.stack(per_beam_log_probs, dim=1)

            if finished.any():
                next_token_log_probs = next_token_log_probs.clone()
                next_token_log_probs[finished] = float('-inf')
                flat = next_token_log_probs.view(-1, vocab_size)
                flat[finished.view(-1), self.tokenizer.eos_id] = 0.0
                next_token_log_probs = flat.view(batch_size, beam_width, vocab_size)

            cum_scores = scores.unsqueeze(-1) + next_token_log_probs

            cum_scores_flat = cum_scores.view(batch_size, -1)
            topk_log_probs, topk_indices = torch.topk(cum_scores_flat, k=beam_width, dim=-1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            gathered = sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, cur_len))
            sequences = torch.cat([gathered, token_indices.unsqueeze(-1)], dim=-1)

            scores = topk_log_probs
            finished = finished.gather(1, beam_indices) | token_indices.eq(self.tokenizer.eos_id)

        alpha = 0.6
        if alpha > 0:
            lengths = sequences.ne(self.tokenizer.pad_id).sum(dim=-1).clamp_min(1)
            norm = ((5 + lengths.float()) ** alpha) / ((5 + 1) ** alpha)
            normed_scores = scores / norm
        else:
            normed_scores = scores

        sorted_scores, order = torch.sort(normed_scores, dim=-1, descending=True)
        sorted_sequences = sequences.gather(1, order.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
        return sorted_sequences, sorted_scores


    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
message.txt
17 KB
