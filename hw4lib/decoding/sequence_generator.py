import torch
import torch.nn as nn
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
        if penalty == 1.0:
            return logits
        
        if logits.dim() == 2:
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
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
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
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

        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        x = x.to(self.device)
        batch_size, init_len = x.shape
        
        # First-step logits over the current context (batch_size, vocab)
        logits0 = self.score_fn(x)
        logits0 = self._apply_repeat_penalty(logits0, x, repeat_penalty)
        if temperature != 1.0:
            logits0 = logits0 / temperature
        log_probs0 = torch.log_softmax(logits0, dim=-1)
        vocab_size = log_probs0.size(-1)

        # pick top-k tokens for each batch as initial beams
        topk_log_probs, next_tokens = torch.topk(log_probs0, k=beam_width, dim=-1)

        # sequences: (batch_size, beam_width, init_len + 1)
        sequences = x.unsqueeze(1).expand(batch_size, beam_width, init_len).clone()
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)
        scores = topk_log_probs.clone()  # (batch_size, beam_width)
        finished = next_tokens.eq(self.tokenizer.eos_id)  # (batch_size, beam_width)

        max_steps = self.max_length - sequences.size(-1)
        for _ in range(max_steps):
            if finished.all():
                break

            cur_len = sequences.size(-1)

            # Compute log-probs per beam-index but batched across the real batch:
            # For k in 0..beam_width-1 call score_fn on sequences[:, k, :] (shape: batch_size, cur_len)
            per_beam_log_probs: List[torch.Tensor] = []
            for k in range(beam_width):
                seqs_k = sequences[:, k, :].to(self.device)  # (batch_size, cur_len)
                logits_k = self.score_fn(seqs_k)  # expects batch_size in first dim
                logits_k = self._apply_repeat_penalty(logits_k, seqs_k, repeat_penalty)
                if temperature != 1.0:
                    logits_k = logits_k / temperature
                log_probs_k = torch.log_softmax(logits_k, dim=-1)  # (batch_size, vocab)
                per_beam_log_probs.append(log_probs_k)

            # stack to (batch_size, beam_width, vocab)
            next_token_log_probs = torch.stack(per_beam_log_probs, dim=1)

            # Handle finished beams: forbid selecting new tokens and allow only EOS to stay
            if finished.any():
                next_token_log_probs = next_token_log_probs.clone()
                # set entire distribution to -inf for finished positions
                next_token_log_probs[finished] = float('-inf')
                # but allow staying on EOS with zero incremental score
                flat = next_token_log_probs.view(-1, vocab_size)
                flat[finished.view(-1), self.tokenizer.eos_id] = 0.0
                next_token_log_probs = flat.view(batch_size, beam_width, vocab_size)

            # cumulative scores: (batch_size, beam_width, vocab)
            cum_scores = scores.unsqueeze(-1) + next_token_log_probs

            # flatten beam and vocab to select top beam_width overall candidates per batch
            cum_scores_flat = cum_scores.view(batch_size, -1)
            topk_log_probs, topk_indices = torch.topk(cum_scores_flat, k=beam_width, dim=-1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            # gather corresponding sequences and append new tokens
            gathered = sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, cur_len))
            sequences = torch.cat([gathered, token_indices.unsqueeze(-1)], dim=-1)

            scores = topk_log_probs
            finished = finished.gather(1, beam_indices) | token_indices.eq(self.tokenizer.eos_id)

        # length normalization / penalty
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
        
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            scores = torch.where(finished, scores, scores + token_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        eos_mask = seq == tokenizer.eos_id
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]
