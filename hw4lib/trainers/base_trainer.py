%%writefile hw4lib/trainers/base_trainer.py
import wandb
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from hw4lib.data.tokenizer import H4Tokenizer
from hw4lib.utils import create_optimizer, create_scheduler
from hw4lib.model import DecoderOnlyTransformer, EncoderDecoderTransformer
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from torchinfo import summary


class BaseTrainer(ABC):
    """
    Base Trainer class that provides common functionality for all trainers.
    """
    def __init__(
            self,
            model: nn.Module,
            tokenizer: H4Tokenizer,
            config: dict,
            run_name: str,
            config_file: str,
            device: Optional[str] = None
    ):
        # 1. Fix WandB Timeout
        os.environ["WANDB_INIT_TIMEOUT"] = "300"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        
        # 2. Initialize optimizer and scheduler immediately
        # This prevents "Optimizer is not initialized" errors in child classes
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(self.optimizer, self.config)

        self.scaler = torch.amp.GradScaler(device=self.device)
        self.use_wandb = config['training'].get('use_wandb', False)
        
        # Initialize experiment directories
        self.expt_root, self.checkpoint_dir, self.attn_dir, self.text_dir, \
        self.best_model_path, self.last_model_path = self._init_experiment(run_name, config_file)

        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_history = []
    
    @abstractmethod
    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Train for one epoch."""
        pass

    @abstractmethod
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        """Validate for one epoch."""
        pass

    @abstractmethod
    def train(self, train_dataloader, val_dataloader):
        """Full training loop."""
        pass

    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluation loop."""
        pass


    def _init_experiment(self, run_name: str, config_file: str):
        """Initialize experiment directories and save initial files."""
        # Create experiment directory
        expt_root = Path(os.getcwd()) / 'expts' / run_name
        expt_root.mkdir(parents=True, exist_ok=True)

        # Copy config
        shutil.copy2(config_file, expt_root / "config.yaml")

        # 3. Robust Model Summary Generation
        # Wrapped in try/except so training never crashes on logging
        try:
            with open(expt_root / "model_arch.txt", "w") as f:
                # Use string check to avoid import/reload mismatch in notebooks
                model_type = type(self.model).__name__
                
                if "DecoderOnly" in model_type:
                    batch_size = self.config['data'].get('batch_size', 8)
                    max_len    = getattr(self.model, 'max_len', 512)
                    input_size = [(batch_size, max_len), (batch_size,)]
                    dtypes     = [torch.long, torch.long]
                    
                    model_summary = summary(
                        self.model,
                        input_size=input_size,
                        dtypes=dtypes,
                        verbose=0
                    )
                    f.write(str(model_summary))

                elif "EncoderDecoder" in model_type:
                    batch_size = self.config['data'].get('batch_size', 8)
                    max_len = 100
                    num_feats = self.config['data']['num_feats']
                    
                    # Create dummy inputs
                    dummy_feats = torch.randn(batch_size, max_len, num_feats).to(self.device)
                    # Use num_classes if available, else default to 100
                    n_classes = getattr(self.model, 'num_classes', 100)
                    dummy_targets = torch.randint(0, n_classes, (batch_size, max_len)).to(self.device)
                    dummy_src_lens = torch.full((batch_size,), max_len, dtype=torch.long).to(self.device)
                    dummy_tgt_lens = torch.full((batch_size,), max_len, dtype=torch.long).to(self.device)

                    input_data = [dummy_feats, dummy_targets, dummy_src_lens, dummy_tgt_lens]
                    
                    model_summary = summary(
                        self.model,
                        input_data=input_data,
                        verbose=0
                    )
                    f.write(str(model_summary))
                else:
                    # Fallback: Write string repr if model type unknown or torchinfo fails
                    f.write(str(self.model))
                    print(f"Warning: Auto-summary skipped for {model_type}. Wrote string representation.")

        except Exception as e:
            print(f"Warning: Could not generate model summary: {e}")
            # Continue execution without crashing

        # Create subdirectories
        checkpoint_dir = expt_root / 'checkpoints'
        attn_dir = expt_root / 'attn'
        text_dir = expt_root / 'text'
        
        checkpoint_dir.mkdir(exist_ok=True)
        attn_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)

        # Define checkpoint paths
        best_model_path = checkpoint_dir / 'checkpoint-best-metric-model.pth'
        last_model_path = checkpoint_dir / 'checkpoint-last-epoch-model.pth'

        # Wandb initialization
        if self.use_wandb:
            run_id = self.config['training'].get('wandb_run_id', None)
            if run_id and run_id.lower() != "none":
                self.wandb_run = wandb.init(
                    project=self.config['training'].get('wandb_project', 'default-project'),
                    id=run_id,
                    resume="must",
                    config=self.config
                )
            else:
                self.wandb_run = wandb.init(
                    project=self.config['training'].get('wandb_project', 'default-project'),
                    config=self.config,
                    name=run_name
                )

        return expt_root, checkpoint_dir, attn_dir, text_dir, best_model_path, last_model_path

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]], step: int):
        """Generic metric logging method."""
        self.training_history.append({
            'epoch': step,
            **metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {}
            for split, split_metrics in metrics.items():
                for metric_name, value in split_metrics.items():
                    wandb_metrics[f'{split}/{metric_name}'] = value
            wandb_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(wandb_metrics, step=step)
        
        # Print metrics with tree structure
        print(f"\n📊 Metrics (Epoch {step}):")
        
        splits = sorted(metrics.keys())
        for i, split in enumerate(splits):
            is_last_split = i == len(splits) - 1
            split_prefix = "└──" if is_last_split else "├──"
            print(f"{split_prefix} {split.upper()}:")
            
            split_metrics = sorted(metrics[split].items())
            for j, (metric_name, value) in enumerate(split_metrics):
                is_last_metric = j == len(split_metrics) - 1
                metric_prefix = "    └──" if is_last_metric else "    ├──"
                if is_last_split:
                    metric_prefix = "    └──" if is_last_metric else "    ├──"
                else:
                    metric_prefix = "│   └──" if is_last_metric else "│   ├──"
                print(f"{metric_prefix} {metric_name}: {value:.4f}")
        
        print("└── TRAINING:")
        print(f"    └── learning_rate: {self.optimizer.param_groups[0]['lr']:.6f}")


    def _save_attention_plot(self, attn_weights: torch.Tensor, epoch: int, attn_type: str = "self"):
        """Save attention weights visualization."""
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, cmap="viridis", cbar=True)
        plt.title(f"Attention Weights - Epoch {epoch}")
        plt.xlabel("Source Sequence")
        plt.ylabel("Target Sequence")
        
        plot_path = os.path.join(self.attn_dir, f"{attn_type}_attention_epoch{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        
        if self.use_wandb:
            wandb.log({f"{attn_type}_attention": wandb.Image(plot_path)}, step=epoch)


    def _save_generated_text(self, text: dict, suffix: str):
        """Save generated text to JSON file."""
        text_path = os.path.join(self.text_dir, f"text_{suffix}.json")
        with open(text_path, "w") as f:
            json.dump(text, f, indent=4)
    
        if self.use_wandb:
            wandb.save(text_path)


    def save_checkpoint(self, filename: str):
        """Save a checkpoint of the model and training state."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        if self.use_wandb:
            wandb.save(str(checkpoint_path))


    def load_checkpoint(self, filename: str):
        """
        Load a checkpoint.
        Attempts to load each component of the checkpoint separately.
        """
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        try:
            # Try safe load first
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Standard load failed, trying without weights_only restriction: {e}")
            # Fallback for complex objects
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        load_status = {}

        # Model
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            load_status['model'] = True
        except Exception as e:
            print(f"Warning: Failed to load model state: {e}")
            load_status['model'] = False

        # Optimizer
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            load_status['optimizer'] = True
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}")
            load_status['optimizer'] = False

        # Scheduler
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                load_status['scheduler'] = True
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")
                load_status['scheduler'] = False

        # Scaler
        try:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            load_status['scaler'] = True
        except Exception as e:
            print(f"Warning: Failed to load scaler state: {e}")
            load_status['scaler'] = False

        # Metrics
        try:
            self.current_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            self.training_history = checkpoint['training_history']
            load_status['training_state'] = True
        except Exception as e:
            print(f"Warning: Failed to load training state: {e}")
            load_status['training_state'] = False

        # Summary
        successful_loads = [k for k, v in load_status.items() if v]
        failed_loads = [k for k, v in load_status.items() if not v]
        
        if not successful_loads:
            raise RuntimeError("Failed to load any checkpoint components")
        
        print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Successfully loaded: {', '.join(successful_loads)}")
        if failed_loads:
            print(f"Failed to load: {', '.join(failed_loads)}")


    def cleanup(self):
        """Cleanup resources."""
        if self.use_wandb and self.wandb_run:
            wandb.finish()