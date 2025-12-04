import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

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
        """
        if penalty == 1.0:
            return logits
        
        batch_size = logits.size(0)
        vocab_size = logits.size(-1)
        
        # Handle beam search shape: (B, Beam, Vocab) -> (B*Beam, Vocab)
        flat_logits = logits.view(-1, vocab_size)
        flat_seqs = sequences.view(-1, sequences.size(-1))
        
        for i in range(flat_seqs.size(0)):
            unique_tokens = torch.unique(flat_seqs[i])
            previous_scores = flat_logits[i, unique_tokens]
            
            # Apply penalty: separate logic for positive/negative scores
            penalized_scores = torch.where(
                previous_scores < 0,
                previous_scores * penalty,
                previous_scores / penalty
            )
            flat_logits[i, unique_tokens] = penalized_scores
            
        return flat_logits.view_as(logits)

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
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

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
        """Generate sequences using greedy search."""
        self._validate_input(x)
        
        current_seqs = x.to(self.device)
        batch_size = current_seqs.size(0)
        
        log_scores = torch.zeros(batch_size, device=self.device)
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        steps_to_gen = self.max_length - current_seqs.size(1)

        for _ in range(steps_to_gen):
            if finished_mask.all():
                break

            logits = self.score_fn(current_seqs)
            logits = self._apply_repeat_penalty(logits, current_seqs, repeat_penalty)
            
            if temperature != 1.0:
                logits = logits / temperature

            log_probs = F.log_softmax(logits, dim=-1)
            next_token_log_probs, next_tokens = torch.max(log_probs, dim=-1)

            log_scores = log_scores + (next_token_log_probs * (~finished_mask).float())
            
            current_seqs = torch.cat([current_seqs, next_tokens.unsqueeze(1)], dim=1)

            just_finished = (next_tokens == self.tokenizer.eos_id)
            finished_mask = finished_mask | just_finished

        return current_seqs, log_scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences using beam search."""
        self._validate_input(x)
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
            
        x = x.to(self.device)
        batch_size, seq_len = x.shape

        sequences = x.unsqueeze(1).repeat(1, beam_width, 1)

        scores = torch.zeros(batch_size, beam_width, device=x.device)
        if beam_width > 1:
            scores[:, 1:] = float('-inf') 

        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=x.device)

        vocab_size = None

        for _ in range(self.max_length - seq_len):
            if finished.all():
                break
                
            # Flatten beams for parallel processing
            flat_inputs = sequences.view(-1, sequences.size(-1))
            
            # Get Logits
            flat_logits = self.score_fn(flat_inputs)
            
            if vocab_size is None:
                vocab_size = flat_logits.size(-1)
                
            logits = flat_logits.view(batch_size, beam_width, -1)
            
            # Apply penalties
            logits = self._apply_repeat_penalty(logits, sequences, repeat_penalty)
            if temperature != 1.0:
                logits = logits / temperature
            
            log_probs = F.log_softmax(logits, dim=-1)

            candidate_scores = scores.unsqueeze(-1) + log_probs
            
            if finished.any():
                finished_expanded = finished.unsqueeze(-1).expand_as(candidate_scores)
                candidate_scores[finished_expanded] = float('-inf')

            # Flatten to (Batch, Beam * Vocab)
            flat_candidates = candidate_scores.view(batch_size, -1)
            
            # Top-K
            best_scores, best_indices = torch.topk(flat_candidates, beam_width, dim=1)
            
            # Recover indices
            beam_indices = best_indices // vocab_size
            token_indices = best_indices % vocab_size

            new_sequences = torch.zeros(
                batch_size, beam_width, sequences.size(-1) + 1, 
                dtype=torch.long, device=x.device
            )
            
            # Update finished status
            new_finished = torch.zeros_like(finished)

            for b in range(batch_size):
                parent_seqs = sequences[b][beam_indices[b]]
                new_tokens = token_indices[b].unsqueeze(1)
                
                new_sequences[b] = torch.cat([parent_seqs, new_tokens], dim=1)

                prev_fin = finished[b][beam_indices[b]]
                is_eos = (token_indices[b] == self.tokenizer.eos_id)
                new_finished[b] = prev_fin | is_eos
            
            sequences = new_sequences
            scores = best_scores
            finished = new_finished

        # Length Penalty & Sorting
        seq_lengths = (sequences != self.tokenizer.pad_id).sum(dim=-1).float()
        alpha = 0.6 
        penalty = ((5 + seq_lengths) ** alpha) / ((5 + 1) ** alpha)
        normed_scores = scores / penalty
        
        # Final Sort
        sorted_indices = torch.argsort(normed_scores, dim=-1, descending=True)
        
        final_seqs = torch.zeros_like(sequences)
        final_scores = torch.zeros_like(scores)
        
        for b in range(batch_size):
            final_seqs[b] = sequences[b][sorted_indices[b]]
            final_scores[b] = normed_scores[b][sorted_indices[b]]
            
        return final_seqs, final_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences using sampling."""
        self._validate_input(x)
        x = x.to(self.device)
        
        batch_size = x.size(0)
        log_scores = torch.zeros(batch_size, device=self.device)
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        steps_to_gen = self.max_length - x.size(1)

        for _ in range(steps_to_gen):
            if finished_mask.all():
                break

            logits = self.score_fn(x)
            filtered_logits = self._filter_logits(logits, temperature, top_k, top_p)
            log_probs = F.log_softmax(filtered_logits, dim=-1)
            
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            log_scores = torch.where(finished_mask, log_scores, log_scores + token_scores)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished_mask = finished_mask | is_eos

        return x, log_scores

    def _validate_input(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError(f"max_length ({self.max_length}) must be >= input sequence length ({x.size(1)})")

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """Post process sequences to remove content after EOS token."""
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu()
            
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                return seq[:eos_indices[0].item() + 1]
            return seq
        
        result_list = []
        for s in seq:
            eos_indices = (s == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                cut_idx = eos_indices[0].item() + 1
                result_list.append(s[:cut_idx])
            else:
                result_list.append(s)
        return result_list
