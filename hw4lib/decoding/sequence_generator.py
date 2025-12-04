import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer


class SequenceGenerator:
    def __init__(self, score_fn: Callable, tokenizer: H4Tokenizer, max_length: int,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    # ----------------------------- #
    #   Shared helper functions     #
    # ----------------------------- #

    def _score_step(self, sequences, repeat_penalty, temperature):
        """Shared logic for: score_fn → repeat penalty → temperature → log_softmax."""
        logits = self.score_fn(sequences)
        logits = self._apply_repeat_penalty(logits, sequences, repeat_penalty)
        if temperature != 1.0:
            logits = logits / temperature
        return torch.log_softmax(logits, dim=-1)

    def _apply_repeat_penalty(self, logits: torch.Tensor, sequences: torch.Tensor, penalty: float):
        if penalty == 1.0:
            return logits

        if logits.dim() == 2:
            for i in range(sequences.size(0)):
                toks = torch.unique(sequences[i])
                pos = logits[i, toks] > 0
                logits[i, toks] = logits[i, toks] / torch.where(
                    pos, torch.full_like(logits[i, toks], penalty),
                    torch.full_like(logits[i, toks], 1.0 / penalty)
                )
        else:
            for b in range(sequences.size(0)):
                for k in range(sequences.size(1)):
                    toks = torch.unique(sequences[b, k])
                    pos = logits[b, k, toks] > 0
                    logits[b, k, toks] = logits[b, k, toks] / torch.where(
                        pos, torch.full_like(logits[b, k, toks], penalty),
                        torch.full_like(logits[b, k, toks], 1.0 / penalty)
                    )
        return logits

    def _filter_logits(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        logits = logits / temperature

        if top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[..., -1:]] = -float("inf")

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_idx = torch.sort(log_probs, descending=True)
            cumprobs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            mask = cumprobs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0

            remove_mask = mask.scatter(dim=-1, index=sorted_idx, src=mask)
            logits[remove_mask] = -float("inf")

        return logits

    # ----------------------------- #
    #       Greedy decoding         #
    # ----------------------------- #

    def generate_greedy(self, x, temperature=1.0, repeat_penalty=1.0):
        if not torch.is_tensor(x) or x.dim() != 2:
            raise ValueError("x must be shape (batch, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length < input length")

        x = x.to(self.device)
        batch = x.size(0)
        scores = torch.zeros(batch, device=x.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            log_probs = self._score_step(x, repeat_penalty, temperature)
            next_tokens = torch.argmax(log_probs, dim=-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            finished |= (next_tokens == self.tokenizer.eos_id)

        return x, scores

    # ----------------------------- #
    #         Beam search           #
    # ----------------------------- #

    def generate_beam(self, x, beam_width, temperature=1.0, repeat_penalty=1.0):
        if not torch.is_tensor(x) or x.dim() != 2:
            raise ValueError("x must be shape (batch, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")

        x = x.to(self.device)
        batch, init_len = x.shape

        # First step: expand beams
        first_log_probs = self._score_step(x, repeat_penalty, temperature)
        topk_scores, topk_tokens = torch.topk(first_log_probs, beam_width, dim=-1)

        sequences = x.unsqueeze(1).expand(batch, beam_width, init_len).clone()
        sequences = torch.cat([sequences, topk_tokens.unsqueeze(-1)], dim=-1)

        scores = topk_scores.clone()
        finished = topk_tokens.eq(self.tokenizer.eos_id)
        vocab = first_log_probs.size(-1)

        # Beam loop
        for _ in range(self.max_length - sequences.size(-1)):
            if finished.all():
                break

            # Compute log_probs for each beam in parallel
            flat_beams = sequences.view(batch * beam_width, -1)
            flat_log_probs = self._score_step(flat_beams, repeat_penalty, temperature)
            log_probs = flat_log_probs.view(batch, beam_width, vocab)

            # Mask finished beams
            if finished.any():
                mask = finished.unsqueeze(-1)
                log_probs = log_probs.masked_fill(mask, -float("inf"))
                log_probs[finished, self.tokenizer.eos_id] = 0.0

            # Expand all beam paths
            candidate_scores = scores.unsqueeze(-1) + log_probs
            flat_scores = candidate_scores.view(batch, -1)

            topk_scores, topk_idx = torch.topk(flat_scores, beam_width, dim=-1)
            beam_idx = topk_idx // vocab
            token_idx = topk_idx % vocab

            # Gather sequences
            gathered = sequences.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))
            sequences = torch.cat([gathered, token_idx.unsqueeze(-1)], dim=-1)

            scores = topk_scores
            finished = finished.gather(1, beam_idx) | token_idx.eq(self.tokenizer.eos_id)

        # Length penalty
        alpha = 0.6
        lengths = sequences.ne(self.tokenizer.pad_id).sum(dim=-1).clamp_min(1)
        norm = ((5 + lengths.float()) ** alpha) / ((5 + 1) ** alpha)
        normed_scores = scores / norm

        sorted_scores, order = torch.sort(normed_scores, dim=-1, descending=True)
        sorted_sequences = sequences.gather(
            1, order.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        )

        return sorted_sequences, sorted_scores

    # ----------------------------- #
    #        Sampling decode        #
    # ----------------------------- #

    def generate_sample(self, x, temperature=1.0, top_k=0, top_p=1.0):
        if not torch.is_tensor(x) or x.dim() != 2:
            raise ValueError("x must be shape (batch, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length < input length")

        batch = x.size(0)
        scores = torch.zeros(batch, device=x.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            logits = self.score_fn(x)
            logits = self._filter_logits(logits, temperature, top_k, top_p)
            log_probs = torch.log_softmax(logits, dim=-1)

            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            finished |= next_tokens.eq(self.tokenizer.eos_id)

        return x, scores

    # ----------------------------- #

    @staticmethod
    def post_process_sequence(seq, tokenizer):
        if seq.dim() == 1:
            eos = (seq == tokenizer.eos_id).nonzero()
            return seq[: eos[0].item() + 1] if len(eos) > 0 else seq

        eos_mask = seq == tokenizer.eos_id
        eos_idx = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        mask = eos_idx.cumsum(dim=1).eq(0) | eos_idx
        return [s[:m.sum()] for s, m in zip(seq, mask)]
