"""
Diagnostics logging utilities for SAC training.

Provides structured logging for:
- Per-update-step losses (CSV)
- Probe state values (JSONL)
- TensorBoard scalars and histograms
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import csv
import json
import os
import logging
import time
import numpy as np
import torch


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics logging."""
    enabled: bool = True
    diagnostics_folder: str = None
    log_losses_every_n_steps: int = 1
    log_probe_values_every_n_steps: int = 100
    log_batch_samples_every_n_steps: int = 100  # Log batch samples every N update steps
    n_batch_samples: int = 3  # Number of transitions to sample from each batch
    tensorboard_enabled: bool = True
    probe_plot_every_n_steps: int = 10000
    policy_snapshot_every_n_steps: int = 10000
    flush_every_n_rows: int = 100
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiagnosticsConfig":
        """Create config from dictionary."""
        if d is None:
            return cls(enabled=False)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CSVScalarLogger:
    """Append-only CSV logger for scalar values."""
    
    def __init__(
        self,
        path: str,
        header_fields: List[str],
        flush_every_n: int = 100,
    ):
        self.path = path
        self.header_fields = header_fields
        self.flush_every_n = flush_every_n
        self._row_count = 0
        self._file = None
        self._writer = None
        self._initialized = False
        self._warned_unknown_fields: set = set()  # Fix #11
        
    def _initialize(self):
        """Initialize file and write header if needed."""
        if self._initialized:
            return
            
        file_exists = os.path.isfile(self.path)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        
        if not file_exists:
            self._writer.writerow(self.header_fields)
            self._file.flush()
            
        self._initialized = True
        
    def log(self, **kwargs):
        """Log a row of scalar values."""
        self._initialize()

        # Fix #11: Warn once about unknown fields to catch bugs when new metrics are added
        unknown = set(kwargs.keys()) - set(self.header_fields) - self._warned_unknown_fields
        if unknown:
            self._warned_unknown_fields.update(unknown)
            logger = logging.getLogger(__name__)
            logger.warning(f"CSVScalarLogger: unknown fields not in header (will be dropped): {unknown}")

        row = [kwargs.get(field, "") for field in self.header_fields]
        self._writer.writerow(row)
        self._row_count += 1
        
        if self._row_count % self.flush_every_n == 0:
            self._file.flush()
            
    def flush(self):
        """Force flush to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


def _convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_serializable(item) for item in obj)
    else:
        return obj


class JSONLLogger:
    """Append-only JSONL logger for structured data (vectors, dicts)."""

    def __init__(
        self,
        path: str,
        flush_every_n: int = 10,
    ):
        self.path = path
        self.flush_every_n = flush_every_n
        self._row_count = 0
        self._file = None
        self._initialized = False
        self._logger = logging.getLogger(__name__)
        
    def _initialize(self):
        """Initialize file."""
        if self._initialized:
            return
        self._file = open(self.path, "a", encoding="utf-8")
        self._initialized = True
        self._logger.info(f"JSONLLogger initialized: {self.path}")
        
    def log(self, data: Dict[str, Any]):
        """Log a JSON object. numpy arrays are converted to lists."""
        self._initialize()

        try:
            serializable = _convert_to_serializable(data)
            json_str = json.dumps(serializable)
            self._file.write(json_str + "\n")
            self._row_count += 1
            
            # Flush immediately for first few rows to ensure data is written
            if self._row_count <= 3 or self._row_count % self.flush_every_n == 0:
                self._file.flush()
                
        except Exception as e:
            self._logger.error(f"JSONLLogger.log failed: {e}, data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
    def flush(self):
        """Force flush to disk."""
        if self._file is not None:
            self._file.flush()
            
    def close(self):
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class TensorBoardLogger:
    """Wrapper around PyTorch's SummaryWriter."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of SummaryWriter."""
        if self._initialized:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
            self._initialized = True
        except ImportError:
            logging.warning("TensorBoard not available. Skipping TensorBoard logging.")
            self._writer = None
            self._initialized = True
            
    def add_scalar(self, tag: str, value: float, global_step: int):
        """Add a scalar value."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step)
            
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """Add multiple scalars under a main tag."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)
            
    def add_histogram(self, tag: str, values: np.ndarray, global_step: int):
        """Add a histogram of values."""
        self._initialize()
        if self._writer is not None:
            self._writer.add_histogram(tag, values, global_step)
            
    def flush(self):
        """Force flush to disk."""
        if self._writer is not None:
            self._writer.flush()
            
    def close(self):
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class DiagnosticsLogger:
    """
    Main diagnostics logger that combines CSV, JSONL, and TensorBoard logging.
    
    This is instantiated in the trainer subprocess and handles all diagnostics logging.
    """
    
    # Standard header fields for loss logging
    LOSS_HEADER_FIELDS = [
        "wall_time",  # Unix timestamp for time correlation
        "update_step",
        "explore_step",
        "q1_loss",
        "q2_loss", 
        "policy_loss",
        "alpha",
        "alpha_loss",
        "log_pi_mean",
        "log_pi_std",
        "entropy_proxy",
        "q1_mean",
        "q2_mean",
        "target_q_mean",
        "min_q_mean",
        "grad_norm_q1",
        "grad_norm_q2",
        "grad_norm_policy",
        "lr_policy",
        # Fix #6: clamp fraction for direct saturation detection
        "clamp_fraction",
        # NEW: Enhanced diagnostics
        "target_entropy",
        "log_alpha",
        # NEW: NaN/Inf sentinels
        "nonfinite_q_loss_count",
        "nonfinite_policy_loss_count",
        "nonfinite_grad_count",
        # Plan v11 risk audit #1 — AWAC advantage-weight saturation
        # diagnostics. Populated only when algo="awac"; SAC runs leave
        # them as 0 (the algo branch never executes _update_policy's
        # AWAC code path). Empty cells become 0 in pandas, which is
        # benign for downstream analysis.
        "awac_weight_saturation",
        "awac_weight_max",
        "awac_weight_mean",
        # RL_IMPROV_16 — E1b weight-spread gate (p99/p1; target 5-20x,
        # ~1.7x = BC-degenerate as in v2) + E2.3 aux-distillation loss
        # visibility (was silently ~0 for the whole v2 run).
        "awac_weight_p99p1",
        "aux_loss",
        # FIX 3 (RL_IMPROV_8 KL-anchor iteration) — IQL publishes the
        # equivalent advantage-weight saturation metrics under the
        # `awr_weight_*` namespace (iql.py last_metrics keys). Before
        # this fix, iql.py emitted these keys but the CSV header only
        # listed `awac_weight_*`, so every IQL update silently dropped
        # the saturation/mean/max values for 100k steps. With these
        # rows added, IQL runs now persist the AWR clamp diagnostics
        # to losses_*.csv and we can verify whether the clamp at
        # iql_awr_max is the rate-limiting factor.
        "awr_weight_saturation",
        "awr_weight_max",
        "awr_weight_mean",
        # FIX 1 (RL_IMPROV_8 KL-anchor iteration) — KL-to-warmstart
        # penalty diagnostics. Populated by IQL._update_policy when
        # alpha_kl > 0 (kl_to_warmstart = closed-form Gaussian KL
        # between current policy and the frozen warm-start reference).
        # 0 when alpha_kl == 0 (no anchor active).
        "kl_to_warmstart",
        "alpha_kl",
    ]
    
    def __init__(
        self,
        config: DiagnosticsConfig,
        process_name: str = "trainer",
    ):
        self.config = config
        self.process_name = process_name
        self.logger = logging.getLogger(__name__)
        
        # Probe states for evaluation (set externally)
        self.probe_states: Optional[torch.Tensor] = None
        self.probe_states_np: Optional[np.ndarray] = None
        
        if not config.enabled or config.diagnostics_folder is None:
            self._loss_logger = None
            self._probe_logger = None
            self._batch_logger = None
            self._tb_logger = None
            self.logger.info("Diagnostics logging disabled")
            return
            
        # Create subdirectories
        csv_dir = os.path.join(config.diagnostics_folder, "csv")
        tb_dir = os.path.join(config.diagnostics_folder, "tensorboard")
        
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        
        # Initialize loggers
        self._loss_logger = CSVScalarLogger(
            path=os.path.join(csv_dir, f"losses_{process_name}.csv"),
            header_fields=self.LOSS_HEADER_FIELDS,
            flush_every_n=config.flush_every_n_rows,
        )
        
        self._probe_logger = JSONLLogger(
            path=os.path.join(csv_dir, f"probe_values_{process_name}.jsonl"),
            flush_every_n=10,
        )
        
        # Batch transitions logger - logs sampled transitions from replay buffer
        self._batch_logger = JSONLLogger(
            path=os.path.join(csv_dir, f"batch_samples_{process_name}.jsonl"),
            flush_every_n=100,
        )
        
        if config.tensorboard_enabled:
            self._tb_logger = TensorBoardLogger(
                log_dir=os.path.join(tb_dir, process_name),
            )
        else:
            self._tb_logger = None
            
        self.logger.info(f"Diagnostics logging initialized for {process_name}")
        
    def log_losses(
        self,
        update_step: int,
        explore_step: int,
        q1_loss: float,
        q2_loss: float,
        policy_loss: float,
        **extra_metrics,
    ):
        """Log loss values and additional metrics."""
        if self._loss_logger is None:
            return
            
        # Check if we should log this step
        if update_step % self.config.log_losses_every_n_steps != 0:
            return
            
        # Build the log entry
        log_data = {
            "wall_time": time.time(),  # Unix timestamp for time correlation
            "update_step": update_step,
            "explore_step": explore_step,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "policy_loss": policy_loss,
        }
        log_data.update(extra_metrics)
        
        self._loss_logger.log(**log_data)
        
        # Also log to TensorBoard
        if self._tb_logger is not None:
            self._tb_logger.add_scalar("losses/q1_loss", q1_loss, update_step)
            self._tb_logger.add_scalar("losses/q2_loss", q2_loss, update_step)
            self._tb_logger.add_scalar("losses/policy_loss", policy_loss, update_step)
            
            for key, value in extra_metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_logger.add_scalar(f"metrics/{key}", value, update_step)
                    
    def log_probe_values(
        self,
        update_step: int,
        probe_values: Dict[str, np.ndarray],
        explore_step: int = None,  # NEW: Added for timestamp correlation
    ):
        """Log probe state evaluation values."""
        if self._probe_logger is None:
            return
            
        log_data = {
            "wall_time": time.time(),  # Unix timestamp for time correlation
            "update_step": update_step,
            "explore_step": explore_step,
            **probe_values,
        }
        self._probe_logger.log(log_data)
        
        # Log summary stats to TensorBoard
        if self._tb_logger is not None:
            for key, values in probe_values.items():
                if isinstance(values, np.ndarray):
                    # Log mean of the first probe state's values
                    if len(values.shape) > 1:
                        first_probe = values[0]
                    else:
                        first_probe = values
                    self._tb_logger.add_scalar(
                        f"probe/{key}_mean", 
                        np.mean(first_probe), 
                        update_step
                    )

    def log_batch_samples(
        self,
        update_step: int,
        explore_step: int,
        batch_samples: List[Dict[str, Any]],
    ):
        """
        Log sampled transitions from the replay batch.
        
        Each sample contains:
        - state (s_t): flattened observation
        - action (a_t): action taken
        - reward (r_t): reward received
        - next_state (s_{t+1}): next observation
        - done: terminal flag
        - actor_values: policy outputs for s_t (mean, log_std, det_action)
        - critic_values: Q1, Q2 for (s_t, a_t) and (s_t, actor_action)
        """
        if self._batch_logger is None:
            self.logger.warning(f"log_batch_samples called but _batch_logger is None at step {update_step}")
            return
            
        try:
            log_data = {
                "wall_time": time.time(),  # Unix timestamp for time correlation
                "update_step": update_step,
                "explore_step": explore_step,
                "n_samples": len(batch_samples),
                "samples": batch_samples,
            }
            self._batch_logger.log(log_data)
            
            # Log success for first few calls
            if update_step <= 300:
                self.logger.info(f"log_batch_samples: logged {len(batch_samples)} samples at step {update_step}")
        except Exception as e:
            self.logger.error(f"log_batch_samples failed at step {update_step}: {e}")
                    
    def set_probe_states(self, probe_states: np.ndarray, device: torch.device = None):
        """Set the probe states for evaluation."""
        self.probe_states_np = probe_states
        if device is not None:
            self.probe_states = torch.as_tensor(
                probe_states, dtype=torch.float32, device=device
            )
        else:
            self.probe_states = torch.as_tensor(
                probe_states, dtype=torch.float32
            )
        self.logger.info(f"Set {len(probe_states)} probe states for evaluation")
        
    def save_probe_states(self, path: str):
        """Save probe states to disk."""
        if self.probe_states_np is not None:
            np.savez(path, probe_states=self.probe_states_np)
            self.logger.info(f"Saved probe states to {path}")
            
    def flush(self):
        """Flush all loggers."""
        if self._loss_logger is not None:
            self._loss_logger.flush()
        if self._probe_logger is not None:
            self._probe_logger.flush()
        if self._batch_logger is not None:
            self._batch_logger.flush()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            
    def close(self):
        """Close all loggers."""
        if self._loss_logger is not None:
            self._loss_logger.close()
        if self._probe_logger is not None:
            self._probe_logger.close()
        if self._batch_logger is not None:
            self._batch_logger.close()
        if self._tb_logger is not None:
            self._tb_logger.close()
