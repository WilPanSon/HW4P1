from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt
from torch.utils.data import Subset, DataLoader
import pandas as pd


class ASRTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss'].get('label_smoothing', 0.0)
        )
        
        self.ctc_criterion = None
        self.ctc_weight = self.config['loss'].get('ctc_weight', 0.0)
        if self.ctc_weight > 0:
            self.ctc_criterion = nn.CTCLoss(
                blank=self.tokenizer.pad_id,
                zero_infinity=True
            )

    def _train_epoch(self, dataloader):
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training ASR]")
        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0
        running_att = None

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch
            
            feats = feats.to(self.device)
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            feat_lengths = feat_lengths.to(self.device)
            transcript_lengths = transcript_lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                seq_out, curr_att, ctc_inputs = self.model(
                    feats, 
                    targets_shifted, 
                    feat_lengths, 
                    transcript_lengths
                )
                
                running_att = curr_att
                
                ce_loss = self.ce_criterion(
                    seq_out.view(-1, seq_out.shape[-1]), 
                    targets_golden.view(-1)
                )
                
                if self.ctc_weight > 0 and ctc_inputs is not None:
                    ctc_loss = self.ctc_criterion(
                        ctc_inputs['log_probs'], 
                        targets_golden, 
                        ctc_inputs['lengths'], 
                        transcript_lengths
                    )
                    loss = ce_loss + self.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0, device=self.device)
                    loss = ce_loss

            batch_tokens = transcript_lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += ce_loss.item() * batch_tokens
            if self.ctc_weight > 0:
                running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens
            
            loss = loss / self.config['training']['gradient_accumulation_steps']

            self.scaler.scale(loss).backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            avg_ce_loss = running_ce_loss / total_tokens if total_tokens > 0 else 0
            avg_ctc_loss = running_ctc_loss / total_tokens if total_tokens > 0 else 0
            avg_joint_loss = running_joint_loss / total_tokens if total_tokens > 0 else 0
            perplexity = torch.exp(torch.tensor(avg_ce_loss))
            
            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}", 
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, curr_att, ctc_inputs, loss
            torch.cuda.empty_cache()

        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_ce_loss = running_ce_loss / total_tokens if total_tokens > 0 else 0
        avg_ctc_loss = running_ctc_loss / total_tokens if total_tokens > 0 else 0
        avg_joint_loss = running_joint_loss / total_tokens if total_tokens > 0 else 0
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()))
        batch_bar.close()

        return {
            'ce_loss': avg_ce_loss,
            'ctc_loss': avg_ctc_loss,
            'joint_loss': avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        results = self.recognize(dataloader, config_name="val_greedy")
        
        references = [r['target'] for r in results]
        hypotheses = [r['generated'] for r in results]
        
        metrics = self._calculate_asr_metrics(references, hypotheses)
        
        return metrics, results
    
    def train(self, train_dataloader, val_dataloader, epochs: int):
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")
        
        self.text_max_len = max(val_dataloader.dataset.text_max_len, train_dataloader.dataset.text_max_len)

        best_val_cer  = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            print(f"\nEpoch {epoch+1}/{self.current_epoch + epochs}")
            
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])
            
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            train_attn_keys = list(train_attn.keys())
            if train_attn_keys: 
                decoder_self_keys  = [k for k in train_attn_keys if 'dec_self' in k]
                decoder_cross_keys = [k for k in train_attn_keys if 'dec_cross' in k]
                
                if decoder_self_keys:
                    first_self_key = decoder_self_keys[0]
                    if first_self_key in train_attn:
                        self._save_attention_plot(train_attn[first_self_key][0], epoch, "decoder_self")
                
                if decoder_cross_keys:
                    last_cross_key = decoder_cross_keys[-1]
                    if last_cross_key in train_attn:
                        self._save_attention_plot(train_attn[last_cross_key][0], epoch, "decoder_cross")
            
            self._save_generated_text(val_results, f'val_epoch_{epoch}')
            
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = val_metrics['cer']
                self.save_checkpoint('checkpoint-best-metric-model.pth') 
                print(f"New best model saved with CER: {best_val_cer:.4f}%")

            self.current_epoch += 1
                

    def evaluate(self, dataloader, max_length: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        recognition_configs = self._get_evaluation_recognition_configs()
        
        eval_results = {}
        for config_name, config in recognition_configs.items():
            try:
                print(f"Evaluating with {config_name} config")
                results = self.recognize(dataloader, config, config_name, max_length)     
                generated = [r['generated'] for r in results]
                results_df = pd.DataFrame(
                    {
                        'id': range(len(generated)),
                        'transcription': generated
                    }
                )
                eval_results[config_name] = results_df
                self._save_generated_text(results, f'test_{config_name}_results')
            except Exception as e:
                print(f"Error evaluating with {config_name} config: {e}")
                continue
        
        return eval_results

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None, config_name: Optional[str] = None, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        if max_length is None and not hasattr(self, 'text_max_len'):
            self.text_max_len = dataloader.dataset.text_max_len if hasattr(dataloader.dataset, 'text_max_len') else 300
        
        if recognition_config is None:
            recognition_config = {
                'num_batches': None, 
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = 'greedy'

        if recognition_config.get('lm_model') is not None:
            recognition_config['lm_model'].eval()
            recognition_config['lm_model'].to(self.device)

        generator = SequenceGenerator(
            score_fn=None, 
            tokenizer=self.tokenizer,
            max_length=max_length if max_length is not None else self.text_max_len,
            device=self.device
        )

        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Recognizing ASR] : {config_name}")
        results = []

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                if len(batch) >= 5:
                    feats, _, targets_golden, feat_lengths, _ = batch
                    targets_golden = targets_golden.to(self.device)
                else: 
                    feats = batch[0]
                    feat_lengths = batch[-1] if len(batch) == 2 else batch[3]
                    targets_golden = None

                feats = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)
                
                encoder_output, pad_mask_src, _, _ = self.model.encode(feats, feat_lengths)
                
                def get_score(x):
                    asr_logits = self.model.score(x, encoder_output, pad_mask_src)
                    
                    if recognition_config.get('lm_model') is not None:
                        lm_logits = recognition_config['lm_model'].score(x)
                        return asr_logits + recognition_config['lm_weight'] * lm_logits
                    
                    return asr_logits
                
                generator.score_fn = get_score

                batch_size = feats.size(0)
                prompts = torch.full((batch_size, 1), self.tokenizer.sos_id, dtype=torch.long, device=self.device)

                if recognition_config.get('beam_width', 1) > 1:
                    try:
                         seqs, scores = generator.generate_beam(
                            prompts,
                            beam_width=recognition_config['beam_width'],
                            temperature=recognition_config.get('temperature', 1.0)
                        )
                         seqs = seqs[:, 0, :]
                         scores = scores[:, 0]
                    except AttributeError:
                        print("Warning: generate_beam not found, falling back to greedy")
                        seqs, scores = generator.generate_greedy(
                            prompts,
                            temperature=recognition_config.get('temperature', 1.0),
                            repeat_penalty=recognition_config.get('repeat_penalty', 1.0)
                        )
                else:
                    seqs, scores = generator.generate_greedy(
                        prompts,
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0)
                    )

                del feats, feat_lengths, encoder_output, pad_mask_src, prompts
                torch.cuda.empty_cache()

                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)
                
                if targets_golden is not None:
                    post_processed_targets = generator.post_process_sequence(targets_golden, self.tokenizer)
                    for j, (pred, target) in enumerate(zip(post_processed_preds, post_processed_targets)):
                        results.append({
                            'target': self.tokenizer.decode(target.tolist(), skip_special_tokens=True),
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })
                else:
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })

                batch_bar.update()

                if recognition_config['num_batches'] is not None and i >= recognition_config['num_batches'] - 1:
                    break

            batch_bar.close()
            return results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None, lm_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
        common_config = {
            'num_batches': None,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        greedy_config = common_config.copy()
        greedy_config.update({
            'beam_width': 1,
        })

        beam_10_config = common_config.copy()
        beam_10_config.update({
            'beam_width': 10,
        })
        
        beam_20_config = common_config.copy()
        beam_20_config.update({
            'beam_width': 20,
        })
        
        return {
            'greedy': greedy_config,
            'beam_10': beam_10_config,
            'beam_20': beam_20_config
        }
        
    def _calculate_asr_metrics(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]]) -> Tuple[float, float, float]:
        wer_metric = tmt.WordErrorRate()
        word_edit_metric = tmt.EditDistance(reduction='mean')
        cer_metric = tmt.CharErrorRate()
        
        word_dist = word_edit_metric(hypotheses, references)
        wer = wer_metric(hypotheses, references)  
        cer = cer_metric(hypotheses, references)  

        return {
            'word_dist': word_dist.item(),
            'wer': wer.item() * 100,
            'cer': cer.item() * 100
        }
    
class ProgressiveTrainer(ASRTrainer):
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.current_stage = 0
        self.all_encoder_layers = list(self.model.enc_layers)
        self.all_decoder_layers = list(self.model.dec_layers)

    def configure_stage(self, stage_config):
        print("\n" + "="*80)
        print(f"Starting Stage: {stage_config['name']}".center(80))
        print("="*80)
        
        print(f"\nConfiguration Details:")
        print(f"├── Data Subset: {stage_config['data_subset']*100:.1f}% of training data")
        print(f"├── Training Epochs: {stage_config['epochs']}")
        print(f"├── Dropout: {stage_config['dropout']}")
        print(f"├── Label Smoothing: {stage_config['label_smoothing']}")
        
        self.model.dropout.p = stage_config['dropout']
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=stage_config['label_smoothing']
        )
        
        encoder_freeze = stage_config.get('encoder_freeze', [])
        decoder_freeze = stage_config.get('decoder_freeze', [])
        
        encoder_active_layers = stage_config['encoder_active_layers']
        if encoder_freeze and len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError(f"Encoder freeze list length ({len(encoder_freeze)}) must match number of active encoder layers ({len(encoder_active_layers)})")
        
        self.model.enc_layers = nn.ModuleList([
            self.all_encoder_layers[i] for i in encoder_active_layers
        ])
        self.model.num_encoder_layers = len(encoder_active_layers)
        
        decoder_active_layers = stage_config['decoder_active_layers']
        if decoder_freeze and len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError(f"Decoder freeze list length ({len(decoder_freeze)}) must match number of active decoder layers ({len(decoder_active_layers)})")
        
        self.model.dec_layers = nn.ModuleList([
            self.all_decoder_layers[i] for i in decoder_active_layers
        ])
        self.model.num_decoder_layers = len(decoder_active_layers)

        frozen_count = 0
        trainable_count = 0
        
        for i, freeze in enumerate(encoder_freeze):
            layer = self.model.enc_layers[i]
            for param in layer.parameters():
                param.requires_grad = not freeze
            if freeze:
                frozen_count += 1
            else:
                trainable_count += 1
                
        for i, freeze in enumerate(decoder_freeze):
            layer = self.model.dec_layers[i]
            for param in layer.parameters():
                param.requires_grad = not freeze
            if freeze:
                frozen_count += 1
            else:
                trainable_count += 1
                
        print(f"├── Frozen Layers: {frozen_count}")
        print(f"└── Trainable Layers: {trainable_count}")
        print("-" * 80 + "\n")

    def get_subset_dataloader(self, dataloader: DataLoader, subset_ratio: float) -> DataLoader:
        if subset_ratio >= 1.0:
            return dataloader
            
        dataset_len = len(dataloader.dataset)
        subset_size = int(dataset_len * subset_ratio)
        indices = list(range(subset_size))
        
        subset = Subset(dataloader.dataset, indices)
        
        return DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=True, 
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory
        )

    def transition_to_full_training(self):
        print("\n" + "="*80)
        print("Transitioning to Full Model Training".center(80))
        print("="*80)
        
        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        print("All layers restored and unfrozen.")
        print("-" * 80 + "\n")

    def progressive_train(self, train_dataloader, val_dataloader):
        stages = self.config.get('stages', [])
        if not stages:
            print("No stages defined, skipping progressive training.")
            return

        for i, stage_config in enumerate(stages):
            self.current_stage = i
            
            self.configure_stage(stage_config)
            
            stage_loader = self.get_subset_dataloader(
                train_dataloader, 
                stage_config['data_subset']
            )
            
            self.train(stage_loader, val_dataloader, epochs=stage_config['epochs'])
            
        self.transition_to_full_training()