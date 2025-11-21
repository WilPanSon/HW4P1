from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from ..utils import create_scheduler
from ..decoding.sequence_generator import SequenceGenerator

class LMTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, 
            label_smoothing=self.config['training'].get('label_smoothing', 0.0)
        )

    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Training LM]")
        running_ce_loss = 0.0
        total_tokens = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            targets_shifted, targets_golden, lengths = batch
            
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            lengths = lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                raw_preds, attn_weights = self.model(targets_shifted, lengths)
                raw_loss = self.criterion(
                    raw_preds.view(-1, raw_preds.size(-1)), 
                    targets_golden.view(-1)
                )
                
            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += raw_loss.item() * batch_tokens

            loss = raw_loss / self.config['training']['gradient_accumulation_steps']
            
            self.scaler.scale(loss).backward()
        
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            avg_ce_loss = running_ce_loss / total_tokens if total_tokens > 0 else 0
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ppl=f"{perplexity_token:.4f}",
            )
            batch_bar.update()

            del targets_shifted, targets_golden, lengths, raw_preds, loss

        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        return {
            'ce_loss_token': avg_ce_loss,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, attn_weights
            
            
    def _validate_epoch(self, dataloader):
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Validating LM]")
        running_ce_loss = 0.0
        total_tokens = 0
        last_attn_weights = None

        for i, batch in enumerate(dataloader):
            targets_shifted, targets_golden, lengths = batch
            
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            lengths = lengths.to(self.device)

            with torch.inference_mode():
                raw_preds, attn_weights = self.model(targets_shifted, lengths)
                loss = self.criterion(
                    raw_preds.view(-1, raw_preds.size(-1)), 
                    targets_golden.view(-1)
                )
                last_attn_weights = attn_weights

            batch_tokens = lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += loss.item() * batch_tokens

            avg_ce_loss = running_ce_loss / total_tokens
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ppl=f"{perplexity_token:.4f}",
            )
            batch_bar.update()

            del targets_shifted, targets_golden, lengths, raw_preds, loss

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ce_loss_char = avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
        batch_bar.close()

        return {
            'ce_loss_token': avg_ce_loss,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, last_attn_weights
        

    def train(self, train_dataloader, val_dataloader, epochs: int):
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")
        
        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            val_metrics, val_attn = self._validate_epoch(val_dataloader)

            gen_results = self.generate(val_dataloader, generation_config=self._get_evaluation_generation_configs()['greedy'])
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['ce_loss_token'])

            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)
            
            if train_attn:
                k = list(train_attn.keys())[0]
                self._save_attention_plot(train_attn[k][0], epoch, "train_self")
            if val_attn:
                k = list(val_attn.keys())[0]
                self._save_attention_plot(val_attn[k][0], epoch, "val_self")

            self._save_generated_text(gen_results, f'val_epoch_{epoch}')
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            val_loss = val_metrics['ce_loss_token']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_metric = val_loss
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def generate(self, dataloader, generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if generation_config is None:
            generation_config = {
                'num_samples': 10,
                'prompt_length': 20,
                'seed': 11785,
                'max_length': self.model.max_len,
                'temperature': 1.0,
                'beam_width': 1,
                'repeat_penalty': 1.0,
                'top_k': 0,
                'top_p': 0.0    
            }

        generator = SequenceGenerator(
            score_fn=lambda x: self.model(x)[0][:, -1, :],
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device
        )

        prompts, originals = dataloader.dataset.sample_prompts(
            num_samples=generation_config.get('num_samples', 10),
            prompt_length=generation_config.get('prompt_length', 10),
            seed=generation_config.get('seed', 11785)
        )
        prompts = prompts.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            if generation_config.get('top_k', 0) > 0 or generation_config.get('top_p', 0) > 0:
                raise NotImplementedError 
            elif generation_config.get('beam_width', 1) > 1:
                raise NotImplementedError 
            else:
                seqs, scores = generator.generate_greedy(
                    prompts,
                    temperature=generation_config.get('temperature', 1.0),
                    repeat_penalty=generation_config.get('repeat_penalty', 1.0)
                )

        processed_seqs = generator.post_process_sequence(seqs, self.tokenizer)

        results = []
        for _, (prompt, seq, score, original) in enumerate(zip(prompts, processed_seqs, scores, originals)):
            results.append({
                'prompt': self.tokenizer.decode(prompt.tolist()),
                'original': self.tokenizer.decode(original[len(prompt):].tolist()),
                'generated': self.tokenizer.decode(seq[len(prompt):].tolist()),
                'score': score.item()
            })

        return results

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        common_config = {
            'num_samples': 50,
            'prompt_length': 10,
            'seed': 11785,
            'max_length': self.model.max_len,
        }
        
        greedy_config = common_config.copy()
        greedy_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 0,
            'top_p': 0.0
        })
        
        beam_config = common_config.copy()
        beam_config.update({
            'temperature': 1.0,
            'beam_width': 10,
            'repeat_penalty': 1.2,
            'top_k': 0,
            'top_p': 0.0
        })

        sample_config = common_config.copy()
        sample_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 10,
            'top_p': 0.95
        })
        
        return {
            'greedy': greedy_config,
            'beam': beam_config,
            'sample': sample_config
        }