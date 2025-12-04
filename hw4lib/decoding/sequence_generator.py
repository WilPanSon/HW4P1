import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable, Union
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:
1. Greedy Search
2. Beam Search
3. Sampling with Filtering
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable[[torch.Tensor], torch.Tensor],
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
                # We update logits in-place carefully
                # If logit > 0: logit /= penalty. If logit < 0: logit *= penalty.
                current_logits = logits[idx, unique_tokens]
                logits[idx, unique_tokens] = torch.where(
                    current_logits > 0,
                    current_logits / penalty,
                    current_logits * penalty
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
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
        """Apply temperature, top-k, and top-p filtering to logits."""
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
        
        x = x.to(self.device)
        batch_size = x.size(0)

        # Initialize tracking variables
        scores = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Generation loop
        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            # 1. Forward pass
            logits = self.score_fn(x)  # (batch_size, vocab_size)

            # 2. Apply penalties
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            
            # 3. Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # 4. Calculate probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # 5. Greedy Selection (Argmax)
            next_token_probs, next_tokens = torch.max(log_probs, dim=-1)

            # 6. Update scores (accumulate log probs)
            # Only update scores for sequences that haven't finished yet
            scores = torch.where(finished, scores, scores + next_token_probs)

            # 7. Append new token
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            # 8. Update finished status
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
        
        x = x.to(self.device)
        batch_size = x.size(0)

        # --- Initialization Step ---
        # Get initial logits for the first token
        logits = self.score_fn(x)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        if temperature != 1.0:
            logits = logits / temperature
        
        log_probs = torch.log_softmax(logits, dim=-1) # (batch_size, vocab_size)
        vocab_size = log_probs.size(-1)

        # Initialize beams with top k tokens from the first step
        top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=-1)
        
        # Create initial beams: (batch_size, beam_width, seq_len + 1)
        # Expand input x to match beam width
        sequences = x.unsqueeze(1).repeat(1, beam_width, 1) 
        sequences = torch.cat([sequences, top_k_indices.unsqueeze(-1)], dim=-1)
        
        # Initialize scores
        scores = top_k_log_probs # (batch_size, beam_width)
        
        # Initialize finished status
        finished = (top_k_indices == self.tokenizer.eos_id)

        # --- Generation Loop ---
        for _ in range(self.max_length - sequences.size(-1)):
            if finished.all():
                break

            # Flatten beam dimension to batch dimension for model input
            # Input: (batch_size * beam_width, current_seq_len)
            inp_flat = sequences.view(-1, sequences.size(-1))
            
            # Forward pass
            logits_flat = self.score_fn(inp_flat)
            
            # Reshape to (batch_size, beam_width, vocab_size)
            logits = logits_flat.view(batch_size, beam_width, vocab_size)

            # Apply penalties
            logits = self._apply_repeat_penalty(logits, sequences, repeat_penalty)
            if temperature != 1.0:
                logits = logits / temperature
                
            next_log_probs = torch.log_softmax(logits, dim=-1)

            # Handling finished beams:
            # If a beam is finished, we force it to choose PAD/EOS with 0 cost, and -inf for everything else
            # This ensures the score doesn't change and it doesn't expand into new tokens
            if finished.any():
                next_log_probs = next_log_probs.clone()
                # Set all probs to -inf for finished beams
                # We iterate to apply mask safely or use advanced indexing
                for b in range(batch_size):
                    for k in range(beam_width):
                        if finished[b, k]:
                            next_log_probs[b, k, :] = float('-inf')
                            # Force EOS/PAD selection with 0 cost (score unchanged)
                            next_log_probs[b, k, self.tokenizer.eos_id] = 0.0

            # Calculate scores for all candidates: (batch_size, beam_width, vocab_size)
            # Add current beam scores to next token probabilities
            candidate_scores = scores.unsqueeze(-1) + next_log_probs
            
            # Flatten to (batch_size, beam_width * vocab_size) to pick top-k across all beams
            candidate_scores_flat = candidate_scores.view(batch_size, -1)
            
            # Select top-k best candidates
            # scores_k: (batch_size, beam_width)
            # indices_k: (batch_size, beam_width)
            scores, indices_k = torch.topk(candidate_scores_flat, beam_width, dim=-1)
            
            # Decode indices to recover beam index and token index
            beam_indices = indices_k // vocab_size  # Which beam did this come from?
            token_indices = indices_k % vocab_size  # What is the new token?

            # Construct new sequences
            # Gather the sequences based on beam_indices
            # sequences: (batch, beam, len)
            beam_indices_expanded = beam_indices.unsqueeze(-1).expand(batch_size, beam_width, sequences.size(-1))
            new_sequences = sequences.gather(1, beam_indices_expanded)
            
            # Append new tokens
            new_sequences = torch.cat([new_sequences, token_indices.unsqueeze(-1)], dim=-1)
            sequences = new_sequences
            
            # Update finished status
            # A sequence is finished if the parent was finished OR the new token is EOS
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
        
        x = x.to(self.device)

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
    def post_process_sequence(seq: Union[torch.Tensor, List[torch.Tensor]], tokenizer: H4Tokenizer) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            Truncated sequence(s)
        """
        def truncate(s):
            # Find all EOS indices
            eos_indices = (s == tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                # Truncate after the first EOS (keeping the EOS token itself)
                return s[:eos_indices[0] + 1]
            return s

        if isinstance(seq, torch.Tensor):
            if seq.dim() == 1:
                return truncate(seq)
            elif seq.dim() == 2:
                # Handle batch of sequences
                return [truncate(s) for s in seq]
            elif seq.dim() == 3:
                # Handle beam output (batch, beam, len)
                processed = []
                for beam_group in seq:
                    processed.append([truncate(s) for s in beam_group])
                return processed
        
        return seq
