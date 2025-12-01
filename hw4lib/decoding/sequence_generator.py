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
        
        # Create a mask of generated tokens
        # Shape: (batch, vocab) or (batch, beam, vocab) depending on logits dims
        batch_size = logits.size(0)
        vocab_size = logits.size(-1)
        
        # Handle beam search shape: (B, Beam, Vocab) -> (B*Beam, Vocab)
        flat_logits = logits.view(-1, vocab_size)
        flat_seqs = sequences.view(-1, sequences.size(-1))
        
        for i in range(flat_seqs.size(0)):
            # Get unique tokens in this sequence
            unique_tokens = torch.unique(flat_seqs[i])
            
            # Apply penalty
            # If score < 0, multiply by penalty (make more negative)
            # If score > 0, divide by penalty (make smaller)
            previous_scores = flat_logits[i, unique_tokens]
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
        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
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
        """
        self._validate_input(x)
        
        current_seqs = x.to(self.device)
        batch_size = current_seqs.size(0)
        
        # Track scores and completion
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
        """
        Generate sequences using beam search.
        """
        self._validate_input(x)
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
            
        x = x.to(self.device)
        batch_size, seq_len = x.shape
        

        logits = self.score_fn(x)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-K initialization (expand batch to beams)
        topk_log_probs, topk_tokens = torch.topk(F.log_softmax(logits, dim=-1), beam_width, dim=-1)
        
        # Prepare Beam Tensors
        # Sequences: (Batch, Beam, Length)
        beam_seqs = x.unsqueeze(1).expand(batch_size, beam_width, seq_len)
        beam_seqs = torch.cat([beam_seqs, topk_tokens.unsqueeze(-1)], dim=-1)
        
        # Scores: (Batch, Beam)
        beam_scores = topk_log_probs
        
        # Finished Mask: (Batch, Beam)
        beam_finished = (topk_tokens == self.tokenizer.eos_id)
        
        vocab_size = logits.size(-1)
        steps_to_gen = self.max_length - seq_len - 1

        # --- Beam Search Loop ---
        for _ in range(steps_to_gen):
            if beam_finished.all():
                break
                
            # Flatten beams into batch dimension for efficient model forward pass
            # (Batch * Beam, Length)
            flat_inputs = beam_seqs.view(-1, beam_seqs.size(-1))
            
            # 1. Model Forward
            flat_logits = self.score_fn(flat_inputs)
            
            # 2. Reshape back to (Batch, Beam, Vocab)
            logits = flat_logits.view(batch_size, beam_width, -1)
            
            # 3. Apply Penalties
            logits = self._apply_repeat_penalty(logits, beam_seqs, repeat_penalty)
            if temperature != 1.0:
                logits = logits / temperature

            next_log_probs = F.log_softmax(logits, dim=-1)

            if beam_finished.any():
                # Expand mask to (Batch, Beam, Vocab)
                mask_expanded = beam_finished.unsqueeze(-1).expand_as(next_log_probs)
                next_log_probs[mask_expanded] = float('-inf')

            candidate_scores = beam_scores.unsqueeze(-1) + next_log_probs

            flat_candidate_scores = candidate_scores.view(batch_size, -1)
            
            topk_scores, topk_indices = torch.topk(flat_candidate_scores, beam_width, dim=-1)

            prev_beam_indices = topk_indices // vocab_size
            # What was the new token?
            new_token_indices = topk_indices % vocab_size
            
            new_beam_seqs = torch.zeros(batch_size, beam_width, beam_seqs.size(-1) + 1, dtype=torch.long, device=self.device)
            
            for b in range(batch_size):
                selected_beams = beam_seqs[b, prev_beam_indices[b]] 
                new_tokens = new_token_indices[b].unsqueeze(-1)
                new_beam_seqs[b] = torch.cat([selected_beams, new_tokens], dim=-1)
            
            beam_seqs = new_beam_seqs
            beam_scores = topk_scores
            

            prev_finished = beam_finished.gather(1, prev_beam_indices)
            new_finished = (new_token_indices == self.tokenizer.eos_id)
            beam_finished = prev_finished | new_finished


        seq_lengths = (beam_seqs != self.tokenizer.pad_id).sum(dim=-1).float()
        # Google NMT length penalty formula
        alpha = 0.6 
        penalty = ((5 + seq_lengths) ** alpha) / ((5 + 1) ** alpha)
        normed_scores = beam_scores / penalty
        
        # Sort by normalized score
        sorted_indices = torch.argsort(normed_scores, dim=-1, descending=True)
        
        # Gather sorted results
        final_seqs = torch.zeros_like(beam_seqs)
        final_scores = torch.zeros_like(beam_scores)
        
        for b in range(batch_size):
            final_seqs[b] = beam_seqs[b, sorted_indices[b]]
            final_scores[b] = normed_scores[b, sorted_indices[b]]
            
        return final_seqs, final_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling.
        """
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
            # Filtering applies temp, top-k, top-p
            filtered_logits = self._filter_logits(logits, temperature, top_k, top_p)
            log_probs = F.log_softmax(filtered_logits, dim=-1)
            
            # Sample from distribution
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Accumulate scores
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            log_scores = torch.where(finished_mask, log_scores, log_scores + token_scores)

            # Update sequence
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # Check EOS
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
        """
        Post process sequences to remove content after EOS token.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu()
            
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                return seq[:eos_indices[0].item() + 1] # Include EOS
            return seq
        
        # Handle batched sequences -> Return list of tensors
        result_list = []
        for s in seq:
            eos_indices = (s == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                cut_idx = eos_indices[0].item() + 1
                result_list.append(s[:cut_idx])
            else:
                result_list.append(s)
        return result_list
