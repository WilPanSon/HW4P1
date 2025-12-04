import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable, Union
from ..data import H4Tokenizer

class SequenceGenerator:
    def __init__(
            self,
            score_fn: Callable[[torch.Tensor], torch.Tensor],
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
                current_logits = logits[idx, unique_tokens]
                logits[idx, unique_tokens] = torch.where(
                    current_logits > 0,
                    current_logits / penalty,
                    current_logits * penalty
                )
        else:
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    current_logits = logits[batch_idx, beam_idx, unique_tokens]
                    logits[batch_idx, beam_idx, unique_tokens] = torch.where(
                        current_logits > 0,
                        current_logits / penalty,
                        current_logits * penalty
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
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

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
        batch_size = x.size(0)

        scores = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            logits = self.score_fn(x)

            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            
            if temperature != 1.0:
                logits = logits / temperature

            log_probs = torch.log_softmax(logits, dim=-1)

            next_token_probs, next_tokens = torch.max(log_probs, dim=-1)

            scores = torch.where(finished, scores, scores + next_token_probs)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
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
        batch_size = x.size(0)

        logits = self.score_fn(x)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        if temperature != 1.0:
            logits = logits / temperature
        
        log_probs = torch.log_softmax(logits, dim=-1)
        vocab_size = log_probs.size(-1)

        top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=-1)
        
        sequences = x.unsqueeze(1).repeat(1, beam_width, 1) 
        sequences = torch.cat([sequences, top_k_indices.unsqueeze(-1)], dim=-1)
        
        scores = top_k_log_probs
        
        finished = (top_k_indices == self.tokenizer.eos_id)

        for _ in range(self.max_length - sequences.size(-1)):
            if finished.all():
                break

            inp_flat = sequences.view(-1, sequences.size(-1))
            
            logits_flat = self.score_fn(inp_flat)
            
            logits = logits_flat.view(batch_size, beam_width, vocab_size)

            logits = self._apply_repeat_penalty(logits, sequences, repeat_penalty)
            if temperature != 1.0:
                logits = logits / temperature
                
            next_log_probs = torch.log_softmax(logits, dim=-1)

            if finished.any():
                next_log_probs = next_log_probs.clone()
                for b in range(batch_size):
                    for k in range(beam_width):
                        if finished[b, k]:
                            next_log_probs[b, k, :] = float('-inf')
                            next_log_probs[b, k, self.tokenizer.eos_id] = 0.0

            candidate_scores = scores.unsqueeze(-1) + next_log_probs
            
            candidate_scores_flat = candidate_scores.view(batch_size, -1)
            
            scores, indices_k = torch.topk(candidate_scores_flat, beam_width, dim=-1)
            
            beam_indices = indices_k // vocab_size
            token_indices = indices_k % vocab_size

            beam_indices_expanded = beam_indices.unsqueeze(-1).expand(batch_size, beam_width, sequences.size(-1))
            new_sequences = sequences.gather(1, beam_indices_expanded)
            
            new_sequences = torch.cat([new_sequences, token_indices.unsqueeze(-1)], dim=-1)
            sequences = new_sequences
            
            prior_finished = finished.gather(1, beam_indices)
            new_finished = (token_indices == self.tokenizer.eos_id)
            finished = prior_finished | new_finished

        return sequences, scores

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
        
        x = x.to(self.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_scores = self.score_fn(x)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: Union[torch.Tensor, List[torch.Tensor]], tokenizer: H4Tokenizer) -> Union[torch.Tensor, List[torch.Tensor]]:
        def truncate(s):
            eos_indices = (s == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                return s[:eos_indices[0] + 1]
            return s

        if isinstance(seq, torch.Tensor):
            if seq.dim() == 1:
                return truncate(seq)
            elif seq.dim() == 2:
                return [truncate(s) for s in seq]
            elif seq.dim() == 3:
                processed = []
                for beam_group in seq:
                    processed.append([truncate(s) for s in beam_group])
                return processed
        
        return seq
