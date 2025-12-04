import torch
from typing import Tuple, Callable
from ..data import H4Tokenizer


class SequenceGenerator:
    def __init__(self, score_fn: Callable, tokenizer: H4Tokenizer,
                 max_length: int, device: str = None):

        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_repeat_penalty(self, logits, sequences, penalty: float):
        if penalty == 1.0:
            return logits

        if logits.dim() == 2:
            for b in range(logits.size(0)):
                toks = torch.unique(sequences[b])
                pos = logits[b, toks] > 0
                logits[b, toks[pos]] /= penalty
                logits[b, toks[~pos]] *= penalty

        else:
            for b in range(logits.size(0)):
                for k in range(logits.size(1)):
                    toks = torch.unique(sequences[b, k])
                    pos = logits[b, k, toks] > 0
                    logits[b, k, toks[pos]] /= penalty
                    logits[b, k, toks[~pos]] *= penalty

        return logits

    def _filter_logits(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        logits = logits / temperature

        if top_k > 0:
            kth_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1:]
            logits[logits < kth_vals] = -float("inf")

        if top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask_sorted = cum > top_p
            mask_sorted[..., 1:] = mask_sorted[..., :-1]
            mask_sorted[..., 0] = 0
            mask = torch.zeros_like(mask_sorted).scatter(-1, sorted_idx, mask_sorted)
            logits[mask] = -float("inf")

        return logits

    def generate_greedy(self, x, temperature=1.0, repeat_penalty=1.0):
        if x.dim() != 2:
            raise ValueError("Input x must have shape (batch, seq_len)")
        if x.size(1) > self.max_length:
            raise ValueError("Input longer than max_length")

        x = x.to(self.device)
        batch = x.size(0)
        scores = torch.zeros(batch, device=x.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            logits = self.score_fn(x)
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            logits /= temperature

            log_probs = torch.log_softmax(logits, dim=-1)
            next_tokens = log_probs.argmax(dim=-1)
            next_scores = log_probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(1)

            scores = torch.where(finished, scores, scores + next_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            finished |= next_tokens.eq(self.tokenizer.eos_id)

        return x, scores

    def generate_beam(self, x, beam_width, temperature=1.0, repeat_penalty=1.0):
        if x.dim() != 2:
            raise ValueError("Input x must have shape (batch, seq_len)")
        if x.size(1) > self.max_length:
            raise ValueError("Input longer than max_length")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")

        x = x.to(self.device)
        batch, init_len = x.shape

        logits = self.score_fn(x)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        logits /= temperature
        log_probs = torch.log_softmax(logits, dim=-1)

        vocab_size = log_probs.size(-1)
        top_scores, next_tokens = torch.topk(log_probs, beam_width, dim=-1)

        sequences = x.unsqueeze(1).repeat(1, beam_width, 1)
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

        beam_scores = top_scores.clone()
        finished = next_tokens.eq(self.tokenizer.eos_id)

        for _ in range(self.max_length - sequences.size(-1)):
            if finished.all():
                break

            step_log_probs = []
            for k in range(beam_width):
                logits_k = self.score_fn(sequences[:, k, :])
                logits_k = self._apply_repeat_penalty(logits_k, sequences[:, k, :], repeat_penalty)
                logits_k /= temperature
                step_log_probs.append(torch.log_softmax(logits_k, dim=-1))

            step_log_probs = torch.stack(step_log_probs, dim=1)

            if finished.any():
                blocked = step_log_probs.clone()
                blocked[finished] = -float("inf")
                blocked[finished, :, self.tokenizer.eos_id] = 0
                step_log_probs = blocked

            total_scores = beam_scores.unsqueeze(-1) + step_log_probs
            flat_scores = total_scores.view(batch, -1)

            new_scores, idx = torch.topk(flat_scores, beam_width, dim=-1)

            new_beam_ids = idx // vocab_size
            new_token_ids = idx % vocab_size

            old_len = sequences.size(-1)
            reordered = sequences.gather(
                1, new_beam_ids.unsqueeze(-1).expand(batch, beam_width, old_len)
            )
            sequences = torch.cat([reordered, new_token_ids.unsqueeze(-1)], dim=-1)

            beam_scores = new_scores
            finished = finished.gather(1, new_beam_ids) | new_token_ids.eq(self.tokenizer.eos_id)

        alpha = 0.6
        lengths = sequences.ne(self.tokenizer.pad_id).sum(dim=-1).float()
        lp = ((5 + lengths)**alpha) / (6**alpha)
        normalized = beam_scores / lp

        final_scores, order = torch.sort(normalized, descending=True, dim=-1)
        final_sequences = sequences.gather(
            1, order.unsqueeze(-1).expand_as(sequences)
        )

        return final_sequences, final_scores

    def generate_sample(self, x, temperature=1.0, top_k=0, top_p=1.0):
        if x.dim() != 2:
            raise ValueError("Input x must have shape (batch, seq_len)")
        if x.size(1) > self.max_length:
            raise ValueError("Input longer than max_length")

        x = x.to(self.device)
        batch = x.size(0)
        scores = torch.zeros(batch, device=x.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            logits = self.score_fn(x)
            logits = self._filter_logits(logits, temperature, top_k, top_p)

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            next_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + next_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            finished |= next_tokens.eq(self.tokenizer.eos_id)

            if finished.all():
                break

        return x, scores

    @staticmethod
    def post_process_sequence(seq, tokenizer):
        if seq.dim() == 1:
            eos_pos = (seq == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            return seq[: eos_pos[0] + 1] if len(eos_pos) else seq

        out = []
        for s in seq:
            eos_pos = (s == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            out.append(s[: eos_pos[0] + 1] if len(eos_pos) else s)
        return out
