%%writefile hw4lib/trainers/asr_trainer.py
from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchmetrics.text as tmt
from torch.utils.data import Subset
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

        if not hasattr(self, "scaler"):
            self.scaler = None

    def _get_time_reduction(self) -> int:
        for attr in ["time_reduction", "time_reduction_factor"]:
            if hasattr(self.model, attr):
                tr = getattr(self.model, attr)
                if isinstance(tr, int) and tr >= 1:
                    return tr

        if hasattr(self.model, "speech_embedding"):
            se = self.model.speech_embedding
            for attr in ["time_reduction", "time_reduction_factor"]:
                if hasattr(se, attr):
                    tr = getattr(se, attr)
                    if isinstance(tr, int) and tr >= 1:
                        return tr

        tr = (
            self.config.get("model", {})
                       .get("speech_embedding", {})
                       .get("time_reduction", 1)
        )
        return int(tr) if isinstance(tr, (int, float)) and tr >= 1 else 1

    def _train_epoch(self, dataloader):
        self.model.train()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="[Training ASR]"
        )

        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0
        running_att = None

        self.optimizer.zero_grad()

        dev_str = str(self.device)
        device_type = "cuda" if "cuda" in dev_str else "cpu"
        amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        time_reduction = self._get_time_reduction()

        for i, batch in enumerate(dataloader):
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch
            feats = feats.to(self.device)
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            feat_lengths = feat_lengths.to(self.device)
            transcript_lengths = transcript_lengths.to(self.device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                seq_out, curr_att, ctc_inputs = self.model(
                    feats, targets_shifted, feat_lengths, transcript_lengths
                )
                running_att = curr_att

                ce_loss = self.ce_criterion(seq_out.permute(0, 2, 1), targets_golden)

                if self.ctc_weight > 0:
                    if isinstance(ctc_inputs, dict) and 'log_probs' in ctc_inputs:
                        log_probs = F.log_softmax(ctc_inputs['log_probs'], dim=-1)
                        ctc_lens = ctc_inputs['lengths']
                    else:
                        log_probs = F.log_softmax(ctc_inputs, dim=-1).transpose(0, 1)
                        ctc_lens = torch.div(
                            feat_lengths, time_reduction, rounding_mode="floor"
                        ).clamp(min=1, max=log_probs.size(0))

                    ctc_loss = self.ctc_criterion(
                        log_probs,
                        targets_golden,
                        ctc_lens,
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

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

                self.optimizer.zero_grad()

            avg_ce_loss = running_ce_loss / total_tokens
            avg_ctc_loss = running_ctc_loss / total_tokens
            avg_joint_loss = running_joint_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_ce_loss))

            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}",
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}"
                         f"/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, curr_att, ctc_inputs, loss
            torch.cuda.empty_cache()

        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
            else:
                self.optimizer.step()
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            self.optimizer.zero_grad()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ctc_loss = running_ctc_loss / total_tokens
        avg_joint_loss = running_joint_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(
            torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token())
        )
        batch_bar.close()

        return {
            'ce_loss': avg_ce_loss,
            'ctc_loss': avg_ctc_loss,
            'joint_loss': avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        full_greedy_cfg = {
            'num_batches': None,
            'beam_width': 1,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': 0.0,
            'lm_model': None
        }

        results = self.recognize(dataloader, recognition_config=full_greedy_cfg, config_name='greedy')

        references = [r['target'] for r in results if 'target' in r]
        hypotheses = [r['generated'] for r in results if 'target' in r]

        metrics = self._calculate_asr_metrics(references, hypotheses)
        return metrics, results

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        recognition_config: Optional[Dict[str, Any]] = None,
        config_name: str = "greedy",
        max_length: Optional[int] = None
    ):
        if max_length is None:
            if hasattr(self, "text_max_len"):
                max_length = self.text_max_len
            elif hasattr(dataloader.dataset, "text_max_len"):
                max_length = dataloader.dataset.text_max_len

        if recognition_config is None:
            recognition_config = {
                'num_batches': None,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = "greedy"

        results = self.recognize(
            dataloader,
            recognition_config=recognition_config,
            config_name=config_name,
            max_length=max_length
        )

        references = [r['target'] for r in results if 'target' in r]
        hypotheses = [r['generated'] for r in results if 'target' in r]

        if references and hypotheses:
            metrics = self._calculate_asr_metrics(references, hypotheses)
        else:
            metrics = {}

        return metrics, results

    def train(self, train_dataloader, val_dataloader, epochs: int):
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")

        self.text_max_len = max(
            val_dataloader.dataset.text_max_len,
            train_dataloader.dataset.text_max_len
        )

        best_val_cer = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])

            metrics = {'train': train_metrics, 'val': val_metrics}
            self._log_metrics(metrics, epoch)

            train_attn_keys = list(train_attn.keys()) if isinstance(train_attn, dict) else []
            if train_attn_keys:
                decoder_self_keys  = [k for k in train_attn_keys if 'dec_self' in k]
                decoder_cross_keys = [k for k in train_attn_keys if 'dec_cross' in k]

                if decoder_self_keys:
                    first_self_key = decoder_self_keys[0]
                    if first_self_key in train_attn:
                        self._save_attention_plot(
                            train_attn[first_self_key][0], epoch, "decoder_self"
                        )

                if decoder_cross_keys:
                    last_cross_key = decoder_cross_keys[-1]
                    if last_cross_key in train_attn:
                        self._save_attention_plot(
                            train_attn[last_cross_key][0], epoch, "decoder_cross"
                        )

            self._save_generated_text(val_results, f'val_epoch_{epoch}')
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = val_metrics['cer']
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def recognize(
        self,
        dataloader,
        recognition_config: Optional[Dict[str, Any]] = None,
        config_name: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if max_length is None and not hasattr(self, 'text_max_len'):
            raise ValueError("text_max_len is not set. Please run training loop first or provide a max_length")

        if recognition_config is None:
            recognition_config = {
                'num_batches': 5,
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
        desc = f"[Recognizing ASR] : {config_name}" if config_name else "[Recognizing ASR]"
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=desc)
        results = []

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                feats, _, targets_golden, feat_lengths, _ = batch
                feats = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)
                if targets_golden is not None:
                    targets_golden = targets_golden.to(self.device)

                encoder_output, pad_mask_src, _, _ = self.model.encode(feats, feat_lengths)

                def get_score(x):
                    asr_logits = self.model.score(x, encoder_output, pad_mask_src)
                    if recognition_config.get('lm_model') is not None:
                        lm_logits = recognition_config['lm_model'].score(x)
                        return asr_logits + recognition_config['lm_weight'] * lm_logits
                    return asr_logits

                generator.score_fn = get_score

                batch_size = feats.size(0)
                prompts = torch.full(
                    (batch_size, 1),
                    self.tokenizer.sos_id,
                    dtype=torch.long,
                    device=self.device
                )

                if recognition_config['beam_width'] > 1 and hasattr(generator, "generate_beam"):
                    seqs, scores = generator.generate_beam(
                        prompts,
                        beam_width=recognition_config.get('beam_width', 1),
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0),
                    )
                    seqs = seqs[:, 0, :]
                    scores = scores[:, 0]
                else:
                    seqs, scores = generator.generate_greedy(
                        prompts,
                        temperature=recognition_config.get('temperature', 1.0),
                        repeat_penalty=recognition_config.get('repeat_penalty', 1.0),
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

    def _calculate_asr_metrics(self, references, hypotheses):
        wer = tmt.WordErrorRate()
        cer = tmt.CharErrorRate()
        
        return {
            'wer': wer(hypotheses, references).item() * 100,
            'cer': cer(hypotheses, references).item() * 100
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

        encoder_active_layers = stage_config['encoder_active_layers']
        decoder_active_layers = stage_config['decoder_active_layers']

        encoder_freeze = stage_config.get('encoder_freeze', None) or [False] * len(encoder_active_layers)
        decoder_freeze = stage_config.get('decoder_freeze', None) or [False] * len(decoder_active_layers)

        if len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError("encoder_freeze must match encoder_active_layers length")
        if len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError("decoder_freeze must match decoder_active_layers length")

        self.model.enc_layers = nn.ModuleList([self.all_encoder_layers[i] for i in encoder_active_layers])
        self.model.dec_layers = nn.ModuleList([self.all_decoder_layers[i] for i in decoder_active_layers])
        self.model.num_encoder_layers = len(encoder_active_layers)
        self.model.num_decoder_layers = len(decoder_active_layers)

        frozen_count = 0
        trainable_count = 0

        print("├── Encoder Layers:")
        for idx, layer in enumerate(self.model.enc_layers):
            should_freeze = encoder_freeze[idx]
            for p in layer.parameters():
                p.requires_grad = not should_freeze
                (frozen_count if should_freeze else trainable_count).__iadd__(p.numel())
            print(f"│   ├── Layer {encoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")

        print("├── Decoder Layers:")
        for idx, layer in enumerate(self.model.dec_layers):
            should_freeze = decoder_freeze[idx]
            for p in layer.parameters():
                p.requires_grad = not should_freeze
                (frozen_count if should_freeze else trainable_count).__iadd__(p.numel())
            print(f"│   ├── Layer {decoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")

        print(f"├── Frozen Parameters: {frozen_count:,}")
        print(f"└── Trainable Parameters: {trainable_count:,}")

    def progressive_train(self, train_dataloader, val_dataloader, stages: List[Dict[str, Any]]):
        for stage_idx, stage_config in enumerate(stages):
            self.current_stage = stage_idx
            self.configure_stage(stage_config)
            subset_train_dataloader = self.get_subset_dataloader(
                train_dataloader, stage_config['data_subset']
            )
            super().train(subset_train_dataloader, val_dataloader, epochs=stage_config['epochs'])

    def transition_to_full_training(self):
        print("\n=== Transitioning to Full Training ===")

        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)

        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss'].get('label_smoothing', 0.0)
        )

        for p in self.model.parameters():
            p.requires_grad = True

        self.best_metric = float('inf')

    def train(self, train_dataloader, val_dataloader, epochs):
        self.transition_to_full_training()
        super().train(train_dataloader, val_dataloader, epochs=epochs)

    def get_subset_dataloader(self, dataloader, subset_fraction):
        dataset = dataloader.dataset
        total_samples = len(dataset)
        subset_size = int(total_samples * subset_fraction)

        indices = torch.randperm(total_samples)[:subset_size]
        subset_dataset = Subset(dataset, indices)

        subset_dataset.text_max_len = dataset.text_max_len
        subset_dataset.feat_max_len = dataset.feat_max_len
        subset_dataset.get_avg_chars_per_token = dataset.get_avg_chars_per_token

        subset_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['NUM_WORKERS'],
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        return subset_loader