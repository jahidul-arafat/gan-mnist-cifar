#!/usr/bin/env python3
"""
Composite Enhanced Metrics Logger for Resume Training - FIXED
============================================================

This enhanced version handles multiple training sessions, properly merging
and continuing logs when resuming from checkpoints.

FIXED: Initialization order issue where attributes were accessed before being defined.

Features:
- Detects existing log files and continues them
- Merges training sessions seamlessly
- Maintains training session history
- Handles epoch numbering correctly across resumes
- Preserves all historical data while adding new data
"""

import json
import os
import time
import glob
from datetime import datetime
from collections import defaultdict
import numpy as np
from pathlib import Path

class CompositeEnhancedMetricsLogger:
    """
    Enhanced metrics logger that handles multiple training sessions and resume scenarios

    Key improvements:
    - Detects and loads existing logs when resuming
    - Maintains session history across multiple runs
    - Properly handles epoch numbering when resuming
    - Merges data from multiple training sessions
    - Preserves all historical training data
    """

    def __init__(self, dataset_key, experiment_name=None, log_dir="./training_logs", resume_from_epoch=None):
        """
        Initialize the composite metrics logger

        Args:
            dataset_key (str): Dataset being used ('mnist' or 'cifar10')
            experiment_name (str, optional): Custom experiment name
            log_dir (str): Directory to save log files
            resume_from_epoch (int, optional): Epoch number we're resuming from
        """
        self.dataset_key = dataset_key
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.resume_from_epoch = resume_from_epoch

        # FIXED: Initialize these attributes first before any method calls
        self.current_epoch = resume_from_epoch or 0
        self.current_step_in_epoch = 0
        self.epoch_start_time = None
        self.current_session_start_epoch = self.current_epoch
        self.current_epoch_steps = []

        # Try to find existing logs for this dataset/experiment
        self.existing_log_found = False
        self.base_experiment_id = None

        # Generate or find experiment identifier
        if resume_from_epoch is not None and resume_from_epoch > 0:
            # We're resuming - try to find existing logs
            existing_log = self._find_existing_log(dataset_key, experiment_name)
            if existing_log:
                self.experiment_id = existing_log["experiment_id"]
                self.base_experiment_id = existing_log["base_id"]
                self.existing_log_found = True
                print(f"📊 Found existing training log: {self.experiment_id}")
                print(f"🔄 Resuming from epoch {resume_from_epoch}")
            else:
                # No existing log found, create new one but mark as resumed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if experiment_name:
                    self.experiment_id = f"{dataset_key}_{experiment_name}_{timestamp}"
                    self.base_experiment_id = f"{dataset_key}_{experiment_name}"
                else:
                    self.experiment_id = f"{dataset_key}_enhanced_dcgan_{timestamp}"
                    self.base_experiment_id = f"{dataset_key}_enhanced_dcgan"
                print(f"⚠️  No existing log found for resume - creating new log: {self.experiment_id}")
        else:
            # Fresh training - create new experiment ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if experiment_name:
                self.experiment_id = f"{dataset_key}_{experiment_name}_{timestamp}"
                self.base_experiment_id = f"{dataset_key}_{experiment_name}"
            else:
                self.experiment_id = f"{dataset_key}_enhanced_dcgan_{timestamp}"
                self.base_experiment_id = f"{dataset_key}_enhanced_dcgan"

        # Define file paths BEFORE trying to load existing data
        self.step_metrics_file = self.log_dir / f"{self.base_experiment_id}_step_metrics.json"
        self.epoch_summaries_file = self.log_dir / f"{self.base_experiment_id}_epoch_summaries.json"
        self.full_training_log = self.log_dir / f"{self.base_experiment_id}_complete_training_log.json"

        # NOW initialize or load existing data (after all attributes are defined)
        if self.existing_log_found:
            self.training_data = self._load_existing_training_data()
            # Add new training session info - NOW safe to call
            self._add_new_training_session()
        else:
            self.training_data = self._initialize_fresh_training_data()

        print(f"📊 Composite Metrics Logger initialized")
        print(f"   🆔 Experiment ID: {self.experiment_id}")
        print(f"   🔗 Base ID: {self.base_experiment_id}")
        print(f"   📁 Log directory: {self.log_dir}")
        print(f"   🔄 Resume from epoch: {self.current_epoch}")
        print(f"   📄 Files:")
        print(f"      • Step metrics: {self.step_metrics_file.name}")
        print(f"      • Epoch summaries: {self.epoch_summaries_file.name}")
        print(f"      • Complete log: {self.full_training_log.name}")

        if self.existing_log_found:
            print(f"   ✅ Loaded existing training data:")
            print(f"      • Previous steps: {len(self.training_data['step_metrics'])}")
            print(f"      • Previous epochs: {len(self.training_data['epoch_summaries'])}")
            print(f"      • Training sessions: {len(self.training_data['training_sessions'])}")

    def _find_existing_log(self, dataset_key, experiment_name):
        """Find existing log files for the same experiment"""

        # Generate possible base experiment IDs
        if experiment_name:
            possible_base_ids = [
                f"{dataset_key}_{experiment_name}",
                f"{dataset_key}_{experiment_name}_enhanced_dcgan"
            ]
        else:
            possible_base_ids = [
                f"{dataset_key}_enhanced_dcgan",
                f"{dataset_key}_dcgan"
            ]

        # Look for existing complete training logs
        for base_id in possible_base_ids:
            pattern = str(self.log_dir / f"{base_id}_complete_training_log.json")
            matching_files = glob.glob(pattern)

            if matching_files:
                # Found existing log
                log_file = matching_files[0]  # Take the first match
                try:
                    with open(log_file, 'r') as f:
                        existing_data = json.load(f)

                    return {
                        "experiment_id": existing_data["experiment_info"]["experiment_id"],
                        "base_id": base_id,
                        "file_path": log_file,
                        "data": existing_data
                    }
                except Exception as e:
                    print(f"⚠️  Found log file but couldn't read it: {e}")
                    continue

        # Also check for partial logs (step metrics or epoch summaries)
        for base_id in possible_base_ids:
            step_pattern = str(self.log_dir / f"{base_id}_step_metrics.json")
            epoch_pattern = str(self.log_dir / f"{base_id}_epoch_summaries.json")

            step_files = glob.glob(step_pattern)
            epoch_files = glob.glob(epoch_pattern)

            if step_files or epoch_files:
                print(f"📁 Found partial logs for {base_id}, will attempt to reconstruct")
                return {
                    "experiment_id": f"{base_id}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "base_id": base_id,
                    "file_path": step_files[0] if step_files else epoch_files[0],
                    "data": None  # Will be reconstructed
                }

        return None

    def _load_existing_training_data(self):
        """Load existing training data and prepare for continuation"""

        # Try to load complete log first
        if self.full_training_log.exists():
            try:
                with open(self.full_training_log, 'r') as f:
                    data = json.load(f)
                print(f"✅ Loaded complete training log")
                return data
            except Exception as e:
                print(f"⚠️  Error loading complete log: {e}")

        # Try to reconstruct from partial files
        print(f"🔄 Reconstructing training data from partial logs...")

        reconstructed_data = self._initialize_fresh_training_data()

        # Load step metrics if available
        if self.step_metrics_file.exists():
            try:
                with open(self.step_metrics_file, 'r') as f:
                    step_data = json.load(f)
                reconstructed_data['step_metrics'] = step_data.get('step_metrics', [])
                print(f"   ✅ Loaded {len(reconstructed_data['step_metrics'])} step metrics")
            except Exception as e:
                print(f"   ⚠️  Error loading step metrics: {e}")

        # Load epoch summaries if available
        if self.epoch_summaries_file.exists():
            try:
                with open(self.epoch_summaries_file, 'r') as f:
                    epoch_data = json.load(f)
                reconstructed_data['epoch_summaries'] = epoch_data.get('epoch_summaries', [])
                print(f"   ✅ Loaded {len(reconstructed_data['epoch_summaries'])} epoch summaries")
            except Exception as e:
                print(f"   ⚠️  Error loading epoch summaries: {e}")

        # Reconstruct epochs list from summaries
        for summary in reconstructed_data['epoch_summaries']:
            epoch_record = {
                "epoch": summary["epoch"],
                "steps_count": summary["step_count"],
                "duration_seconds": summary["duration_seconds"],
                "end_time": summary["timestamp"],
                "summary": summary
            }
            reconstructed_data['epochs'].append(epoch_record)

        return reconstructed_data

    def _initialize_fresh_training_data(self):
        """Initialize fresh training data structure"""
        return {
            "experiment_info": {
                "experiment_id": self.experiment_id,
                "base_experiment_id": self.base_experiment_id,
                "dataset": self.dataset_key,
                "start_time": datetime.now().isoformat(),
                "framework": "Enhanced DCGAN with WGAN-GP",
                "features": [
                    "WGAN-GP Loss",
                    "EMA Generator",
                    "Spectral Normalization",
                    "Adaptive Gradient Penalty",
                    "Progressive Learning Rate",
                    "Enhanced Architecture",
                    "Composite Metrics Logging"
                ]
            },
            "training_config": {},
            "epochs": [],
            "step_metrics": [],
            "epoch_summaries": [],
            "training_events": [],
            "training_sessions": []  # Track multiple training sessions
        }

    def _add_new_training_session(self):
        """Add information about the new training session"""
        session_info = {
            "session_id": len(self.training_data["training_sessions"]) + 1,
            "start_time": datetime.now().isoformat(),
            "resume_from_epoch": self.resume_from_epoch,
            "session_experiment_id": self.experiment_id,
            "previous_total_steps": len(self.training_data["step_metrics"]),
            "previous_total_epochs": len(self.training_data["epoch_summaries"])
        }

        self.training_data["training_sessions"].append(session_info)

        # Log the resume event
        self.log_event("training_session_resumed", f"New training session started (resume from epoch {self.resume_from_epoch})", {
            "session_id": session_info["session_id"],
            "previous_sessions": len(self.training_data["training_sessions"]) - 1
        })

    def set_training_config(self, config_dict):
        """Set or update training configuration parameters"""
        if "training_config" not in self.training_data:
            self.training_data["training_config"] = {}

        # If this is a resumed session, store config as session-specific
        if self.existing_log_found:
            session_id = len(self.training_data["training_sessions"])
            session_config_key = f"session_{session_id}_config"
            self.training_data["training_config"][session_config_key] = config_dict
        else:
            self.training_data["training_config"].update(config_dict)

        self.log_event("training_config_updated", "Training configuration parameters updated")

    def set_system_info(self, device_info, model_info=None):
        """Set or update system and device information"""
        if "system_info" not in self.training_data["experiment_info"]:
            self.training_data["experiment_info"]["system_info"] = {}

        # Always update system info (might have changed between sessions)
        self.training_data["experiment_info"]["system_info"].update(device_info)

        if model_info:
            if "model_info" not in self.training_data["experiment_info"]:
                self.training_data["experiment_info"]["model_info"] = {}
            self.training_data["experiment_info"]["model_info"].update(model_info)

        self.log_event("system_info_updated", f"System info updated for device: {device_info.get('device_type', 'unknown')}")

    def start_epoch(self, epoch_num, total_epochs, total_steps):
        """Mark the start of a new training epoch"""

        # Check if this epoch already exists (shouldn't happen in normal resume, but safety check)
        existing_epoch_idx = None
        for i, epoch_summary in enumerate(self.training_data["epoch_summaries"]):
            if epoch_summary["epoch"] == epoch_num:
                existing_epoch_idx = i
                break

        if existing_epoch_idx is not None:
            print(f"⚠️  Epoch {epoch_num} already exists in logs - this suggests overlapping training")
            # Remove the existing epoch to avoid duplicates
            del self.training_data["epoch_summaries"][existing_epoch_idx]
            # Also remove corresponding step metrics
            self.training_data["step_metrics"] = [
                step for step in self.training_data["step_metrics"]
                if step["epoch"] != epoch_num
            ]

        # Save previous epoch if exists
        if self.current_epoch > 0 and self.current_epoch_steps:
            self.end_epoch()

        self.current_epoch = epoch_num
        self.current_step_in_epoch = 0
        self.epoch_start_time = time.time()
        self.current_epoch_steps = []

        epoch_info = {
            "epoch": epoch_num,
            "total_epochs": total_epochs,
            "total_steps": total_steps,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": self.epoch_start_time,
            "session_id": len(self.training_data["training_sessions"]) if self.training_data["training_sessions"] else 1
        }

        self.log_event("epoch_start", f"Started epoch {epoch_num}/{total_epochs} with {total_steps} steps")
        print(f"📊 Metrics Logger: Started tracking epoch {epoch_num}/{total_epochs}")

    def log_step_metrics(self, step_metrics):
        """Log metrics for a single training step with proper global step numbering"""

        self.current_step_in_epoch += 1
        current_time = time.time()

        # Calculate global step number (across all sessions)
        global_step = len(self.training_data["step_metrics"]) + 1

        # Create comprehensive step record
        step_record = {
            "experiment_id": self.experiment_id,
            "base_experiment_id": self.base_experiment_id,
            "session_id": len(self.training_data["training_sessions"]) if self.training_data["training_sessions"] else 1,
            "epoch": self.current_epoch,
            "step": self.current_step_in_epoch,
            "global_step": global_step,
            "timestamp": datetime.now().isoformat(),
            "time_since_epoch_start": current_time - self.epoch_start_time if self.epoch_start_time else 0,

            # Core training metrics
            "losses": {
                "discriminator_loss": float(step_metrics.get("d_loss", 0)),
                "generator_loss": float(step_metrics.get("g_loss", 0)),
                "wasserstein_distance": float(step_metrics.get("wd", 0)),
                "gradient_penalty": float(step_metrics.get("gp", 0))
            },

            # WGAN-GP specific metrics
            "wgan_gp": {
                "gradient_norm": float(step_metrics.get("grad_norm", 0)),
                "current_lambda_gp": float(step_metrics.get("lambda_gp", 10.0)),
                "target_grad_norm_min": 0.8,
                "target_grad_norm_max": 1.2
            },

            # Model quality metrics
            "quality": {
                "ema_quality_score": float(step_metrics.get("ema_quality", 0)),
                "discriminator_convergence": self._assess_discriminator_convergence(step_metrics),
                "generator_convergence": self._assess_generator_convergence(step_metrics)
            },

            # Training parameters
            "learning_rates": {
                "generator_lr": float(step_metrics.get("lr_g", 0)),
                "discriminator_lr": float(step_metrics.get("lr_d", 0))
            },

            # Performance metrics
            "performance": {
                "batch_time_seconds": float(step_metrics.get("batch_time", 0)),
                "steps_per_second": 1.0 / float(step_metrics.get("batch_time", 1)) if step_metrics.get("batch_time", 0) > 0 else 0,
                "memory_usage": step_metrics.get("memory_usage", "N/A")
            },

            # Health indicators
            "health_indicators": self._calculate_health_indicators(step_metrics)
        }

        # Add to current epoch steps
        self.current_epoch_steps.append(step_record)

        # Add to global step metrics
        self.training_data["step_metrics"].append(step_record)

        # Save step metrics incrementally (every 10 steps)
        if self.current_step_in_epoch % 10 == 0:
            self._save_step_metrics_incremental()

    def end_epoch(self):
        """Mark the end of current epoch and compute epoch summary"""
        if not self.current_epoch_steps:
            return

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time if self.epoch_start_time else 0

        # Calculate epoch-level aggregated metrics
        epoch_summary = self._calculate_epoch_summary(
            self.current_epoch,
            self.current_epoch_steps,
            epoch_duration
        )

        # Add session information to epoch summary
        epoch_summary["session_id"] = len(self.training_data["training_sessions"]) if self.training_data["training_sessions"] else 1
        epoch_summary["cumulative_epochs"] = len(self.training_data["epoch_summaries"]) + 1
        epoch_summary["cumulative_steps"] = len(self.training_data["step_metrics"])

        # Add to epoch summaries
        self.training_data["epoch_summaries"].append(epoch_summary)

        # Add epoch to epochs list
        epoch_record = {
            "epoch": self.current_epoch,
            "steps_count": len(self.current_epoch_steps),
            "duration_seconds": epoch_duration,
            "end_time": datetime.now().isoformat(),
            "session_id": len(self.training_data["training_sessions"]) if self.training_data["training_sessions"] else 1,
            "summary": epoch_summary
        }
        self.training_data["epochs"].append(epoch_record)

        self.log_event("epoch_end", f"Completed epoch {self.current_epoch} with {len(self.current_epoch_steps)} steps")

        # Save epoch summary
        self._save_epoch_summaries()

        print(f"📊 Metrics Logger: Completed epoch {self.current_epoch}")
        print(f"   📈 Average D-Loss: {epoch_summary['losses']['avg_discriminator_loss']:.6f}")
        print(f"   📈 Average G-Loss: {epoch_summary['losses']['avg_generator_loss']:.6f}")
        print(f"   📏 Average Grad Norm: {epoch_summary['wgan_gp']['avg_gradient_norm']:.6f}")
        print(f"   ⏱️  Epoch Duration: {epoch_duration:.1f}s")
        print(f"   🔢 Cumulative Epochs: {epoch_summary['cumulative_epochs']}")
        print(f"   📊 Cumulative Steps: {epoch_summary['cumulative_steps']}")

    def log_event(self, event_type, description, additional_data=None):
        """Log a training event with session information"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "step": self.current_step_in_epoch,
            "session_id": len(self.training_data["training_sessions"]) if self.training_data["training_sessions"] else 1,
            "event_type": event_type,
            "description": description
        }

        if additional_data:
            event["data"] = additional_data

        self.training_data["training_events"].append(event)

    def finalize_training(self):
        """Finalize training and save complete log"""
        # End current epoch if in progress
        if self.current_epoch_steps:
            self.end_epoch()

        # Update current session info
        if self.training_data["training_sessions"]:
            current_session = self.training_data["training_sessions"][-1]
            current_session["end_time"] = datetime.now().isoformat()
            current_session["epochs_completed"] = self.current_epoch - current_session["resume_from_epoch"]
            current_session["steps_completed"] = len(self.training_data["step_metrics"]) - current_session["previous_total_steps"]

        # Update experiment info
        self.training_data["experiment_info"]["end_time"] = datetime.now().isoformat()
        self.training_data["experiment_info"]["total_epochs_completed"] = len(self.training_data["epoch_summaries"])
        self.training_data["experiment_info"]["total_steps_completed"] = len(self.training_data["step_metrics"])
        self.training_data["experiment_info"]["total_training_sessions"] = len(self.training_data["training_sessions"])

        # Calculate overall training statistics
        self.training_data["training_statistics"] = self._calculate_composite_training_statistics()

        # Save complete log
        self._save_complete_log()

        self.log_event("training_complete", "Training session completed and logs finalized")

        print(f"📊 Composite Metrics Logger: Training completed!")
        print(f"   📁 Complete log saved: {self.full_training_log}")
        print(f"   📈 Total epochs: {len(self.training_data['epoch_summaries'])}")
        print(f"   📊 Total steps: {len(self.training_data['step_metrics'])}")
        print(f"   🔄 Training sessions: {len(self.training_data['training_sessions'])}")

    def _calculate_composite_training_statistics(self):
        """Calculate statistics across all training sessions"""
        if not self.training_data["epochs"]:
            return {}

        all_epochs = self.training_data["epoch_summaries"]
        all_sessions = self.training_data["training_sessions"]

        # Overall performance across all sessions
        total_steps = len(self.training_data["step_metrics"])
        total_duration = sum(epoch["duration_seconds"] for epoch in all_epochs)

        # Session-specific statistics
        session_stats = []
        for session in all_sessions:
            session_epochs = [
                epoch for epoch in all_epochs
                if epoch.get("session_id") == session["session_id"]
            ]

            if session_epochs:
                session_stat = {
                    "session_id": session["session_id"],
                    "epochs_in_session": len(session_epochs),
                    "session_duration": sum(epoch["duration_seconds"] for epoch in session_epochs),
                    "session_start_epoch": min(epoch["epoch"] for epoch in session_epochs),
                    "session_end_epoch": max(epoch["epoch"] for epoch in session_epochs),
                }
                session_stats.append(session_stat)

        # Loss trends across all sessions
        epoch_avg_d_losses = [epoch["losses"]["avg_discriminator_loss"] for epoch in all_epochs]
        epoch_avg_g_losses = [epoch["losses"]["avg_generator_loss"] for epoch in all_epochs]
        epoch_avg_wds = [epoch["losses"]["avg_wasserstein_distance"] for epoch in all_epochs]

        return {
            "total_training_time_seconds": total_duration,
            "total_training_time_hours": total_duration / 3600,
            "total_epochs": len(all_epochs),
            "total_steps": total_steps,
            "total_sessions": len(all_sessions),
            "avg_steps_per_epoch": total_steps / len(all_epochs) if all_epochs else 0,
            "avg_epoch_duration": total_duration / len(all_epochs) if all_epochs else 0,

            # Session statistics
            "session_statistics": session_stats,

            # Overall trends (across all sessions)
            "overall_trends": {
                "discriminator_loss_trend": "improving" if epoch_avg_d_losses[-1] < epoch_avg_d_losses[0] else "worsening",
                "generator_loss_trend": "improving" if epoch_avg_g_losses[-1] < epoch_avg_g_losses[0] else "worsening",
                "wasserstein_distance_trend": "improving" if abs(epoch_avg_wds[-1]) < abs(epoch_avg_wds[0]) else "worsening",
                "overall_convergence": "good" if (abs(epoch_avg_wds[-1]) < abs(epoch_avg_wds[0])) else "needs_attention"
            },

            # Final metrics (from last session)
            "final_metrics": {
                "final_discriminator_loss": epoch_avg_d_losses[-1] if epoch_avg_d_losses else 0,
                "final_generator_loss": epoch_avg_g_losses[-1] if epoch_avg_g_losses else 0,
                "final_wasserstein_distance": epoch_avg_wds[-1] if epoch_avg_wds else 0,
                "best_wasserstein_distance": min(epoch_avg_wds, key=abs) if epoch_avg_wds else 0
            },

            # Resume information
            "resume_summary": {
                "total_resumes": len(all_sessions) - 1,
                "longest_session_epochs": max((stat["epochs_in_session"] for stat in session_stats), default=0),
                "average_session_length": np.mean([stat["epochs_in_session"] for stat in session_stats]) if session_stats else 0
            }
        }

    # Helper methods (same as before, but adapted for composite logging)
    def _assess_discriminator_convergence(self, metrics):
        """Assess discriminator convergence based on current metrics"""
        d_loss = abs(float(metrics.get("d_loss", 0)))
        if d_loss < 0.5:
            return "excellent"
        elif d_loss < 1.0:
            return "good"
        elif d_loss < 2.0:
            return "fair"
        else:
            return "poor"

    def _assess_generator_convergence(self, metrics):
        """Assess generator convergence based on current metrics"""
        g_loss = abs(float(metrics.get("g_loss", 0)))
        if g_loss < 0.5:
            return "excellent"
        elif g_loss < 1.0:
            return "good"
        elif g_loss < 2.0:
            return "fair"
        else:
            return "poor"

    def _calculate_health_indicators(self, metrics):
        """Calculate overall health indicators for current step"""
        indicators = {}

        # Gradient norm health (should be close to 1.0 for WGAN-GP)
        grad_norm = float(metrics.get("grad_norm", 0))
        if 0.8 <= grad_norm <= 1.2:
            indicators["gradient_norm_health"] = "optimal"
        elif 0.7 <= grad_norm <= 1.3:
            indicators["gradient_norm_health"] = "good"
        else:
            indicators["gradient_norm_health"] = "poor"

        # Wasserstein distance health (closer to 0 is better)
        wd = abs(float(metrics.get("wd", 0)))
        if wd < 0.5:
            indicators["wasserstein_health"] = "excellent"
        elif wd < 1.0:
            indicators["wasserstein_health"] = "good"
        elif wd < 2.0:
            indicators["wasserstein_health"] = "fair"
        else:
            indicators["wasserstein_health"] = "poor"

        # Overall training health score (0-100)
        health_score = 0
        if indicators["gradient_norm_health"] == "optimal":
            health_score += 40
        elif indicators["gradient_norm_health"] == "good":
            health_score += 25

        if indicators["wasserstein_health"] == "excellent":
            health_score += 30
        elif indicators["wasserstein_health"] == "good":
            health_score += 20
        elif indicators["wasserstein_health"] == "fair":
            health_score += 10

        # EMA quality contribution
        ema_quality = float(metrics.get("ema_quality", 0))
        health_score += min(30, ema_quality * 30)

        indicators["overall_health_score"] = min(100, health_score)

        return indicators

    def _calculate_epoch_summary(self, epoch_num, steps, duration):
        """Calculate comprehensive epoch summary statistics"""
        if not steps:
            return {}

        # Extract metrics arrays
        d_losses = [step["losses"]["discriminator_loss"] for step in steps]
        g_losses = [step["losses"]["generator_loss"] for step in steps]
        wds = [step["losses"]["wasserstein_distance"] for step in steps]
        gps = [step["losses"]["gradient_penalty"] for step in steps]
        grad_norms = [step["wgan_gp"]["gradient_norm"] for step in steps]
        ema_qualities = [step["quality"]["ema_quality_score"] for step in steps]
        batch_times = [step["performance"]["batch_time_seconds"] for step in steps]

        summary = {
            "epoch": epoch_num,
            "step_count": len(steps),
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),

            # Loss statistics
            "losses": {
                "avg_discriminator_loss": float(np.mean(d_losses)),
                "std_discriminator_loss": float(np.std(d_losses)),
                "min_discriminator_loss": float(np.min(d_losses)),
                "max_discriminator_loss": float(np.max(d_losses)),

                "avg_generator_loss": float(np.mean(g_losses)),
                "std_generator_loss": float(np.std(g_losses)),
                "min_generator_loss": float(np.min(g_losses)),
                "max_generator_loss": float(np.max(g_losses)),

                "avg_wasserstein_distance": float(np.mean(wds)),
                "std_wasserstein_distance": float(np.std(wds)),
                "min_wasserstein_distance": float(np.min(wds)),
                "max_wasserstein_distance": float(np.max(wds)),

                "avg_gradient_penalty": float(np.mean(gps)),
                "std_gradient_penalty": float(np.std(gps)),
                "min_gradient_penalty": float(np.min(gps)),
                "max_gradient_penalty": float(np.max(gps))
            },

            # WGAN-GP specific metrics
            "wgan_gp": {
                "avg_gradient_norm": float(np.mean(grad_norms)),
                "std_gradient_norm": float(np.std(grad_norms)),
                "min_gradient_norm": float(np.min(grad_norms)),
                "max_gradient_norm": float(np.max(grad_norms)),
                "gradient_norm_stability": float(1.0 / (1.0 + np.std(grad_norms))),
                "target_adherence_ratio": float(np.mean([(0.8 <= gn <= 1.2) for gn in grad_norms]))
            },

            # Quality metrics
            "quality": {
                "avg_ema_quality": float(np.mean(ema_qualities)),
                "final_ema_quality": float(ema_qualities[-1]) if ema_qualities else 0,
                "ema_improvement": float(ema_qualities[-1] - ema_qualities[0]) if len(ema_qualities) > 1 else 0
            },

            # Performance metrics
            "performance": {
                "avg_batch_time": float(np.mean(batch_times)),
                "std_batch_time": float(np.std(batch_times)),
                "min_batch_time": float(np.min(batch_times)),
                "max_batch_time": float(np.max(batch_times)),
                "avg_steps_per_second": float(1.0 / np.mean(batch_times)) if np.mean(batch_times) > 0 else 0,
                "total_compute_time": float(np.sum(batch_times))
            },

            # Health assessment
            "health_assessment": self._calculate_epoch_health_assessment(steps)
        }

        return summary

    def _calculate_epoch_health_assessment(self, steps):
        """Calculate epoch-level health assessment"""
        if not steps:
            return {}

        health_scores = [step["health_indicators"]["overall_health_score"] for step in steps]
        gradient_healths = [step["health_indicators"]["gradient_norm_health"] for step in steps]
        wasserstein_healths = [step["health_indicators"]["wasserstein_health"] for step in steps]

        return {
            "avg_health_score": float(np.mean(health_scores)),
            "min_health_score": float(np.min(health_scores)),
            "max_health_score": float(np.max(health_scores)),
            "health_trend": "improving" if health_scores[-1] > health_scores[0] else "declining",
            "gradient_norm_optimal_ratio": float(gradient_healths.count("optimal") / len(gradient_healths)),
            "wasserstein_excellent_ratio": float(wasserstein_healths.count("excellent") / len(wasserstein_healths)),
            "overall_stability": float(1.0 / (1.0 + np.std(health_scores)))
        }

    def _save_step_metrics_incremental(self):
        """Save step metrics incrementally with resume-safe structure"""
        try:
            step_data = {
                "experiment_id": self.experiment_id,
                "base_experiment_id": self.base_experiment_id,
                "last_updated": datetime.now().isoformat(),
                "total_steps": len(self.training_data["step_metrics"]),
                "total_sessions": len(self.training_data["training_sessions"]),
                "step_metrics": self.training_data["step_metrics"]
            }

            with open(self.step_metrics_file, 'w') as f:
                json.dump(step_data, f, indent=2, default=str)

        except Exception as e:
            print(f"⚠️  Failed to save step metrics: {e}")

    def _save_epoch_summaries(self):
        """Save epoch summaries with session information"""
        try:
            epoch_data = {
                "experiment_id": self.experiment_id,
                "base_experiment_id": self.base_experiment_id,
                "last_updated": datetime.now().isoformat(),
                "total_epochs": len(self.training_data["epoch_summaries"]),
                "total_sessions": len(self.training_data["training_sessions"]),
                "epoch_summaries": self.training_data["epoch_summaries"],
                "training_sessions": self.training_data["training_sessions"]
            }

            with open(self.epoch_summaries_file, 'w') as f:
                json.dump(epoch_data, f, indent=2, default=str)

        except Exception as e:
            print(f"⚠️  Failed to save epoch summaries: {e}")

    def _save_complete_log(self):
        """Save complete training log with all session data"""
        try:
            with open(self.full_training_log, 'w') as f:
                json.dump(self.training_data, f, indent=2, default=str)

            print(f"✅ Complete composite training log saved: {self.full_training_log}")

        except Exception as e:
            print(f"❌ Failed to save complete training log: {e}")

    def get_current_stats(self):
        """Get current training statistics including session information"""
        if not self.training_data["step_metrics"]:
            return {}

        latest_step = self.training_data["step_metrics"][-1]
        current_session = self.training_data["training_sessions"][-1] if self.training_data["training_sessions"] else None

        return {
            "current_epoch": self.current_epoch,
            "current_step": self.current_step_in_epoch,
            "total_steps_completed": len(self.training_data["step_metrics"]),
            "total_epochs_completed": len(self.training_data["epoch_summaries"]),
            "current_session_id": current_session["session_id"] if current_session else 1,
            "total_sessions": len(self.training_data["training_sessions"]),
            "latest_metrics": latest_step,
            "health_score": latest_step["health_indicators"]["overall_health_score"]
        }

    def get_session_summary(self):
        """Get summary of all training sessions"""
        sessions = self.training_data["training_sessions"]
        if not sessions:
            return "No training sessions recorded"

        summary = f"📊 Training Sessions Summary:\n"
        for session in sessions:
            start_epoch = session.get("resume_from_epoch", 0)
            end_time = session.get("end_time", "ongoing")
            epochs_completed = session.get("epochs_completed", "unknown")
            steps_completed = session.get("steps_completed", "unknown")

            summary += f"   Session {session['session_id']}: "
            summary += f"Started from epoch {start_epoch}, "
            summary += f"completed {epochs_completed} epochs, {steps_completed} steps\n"
            if end_time != "ongoing":
                summary += f"      Ended: {end_time}\n"
            else:
                summary += f"      Status: Currently running\n"

        return summary

# Helper function to create composite metrics logger
def create_composite_metrics_logger(dataset_key, device_info, training_config, experiment_name=None, resume_from_epoch=None):
    """
    Create and configure a composite metrics logger that handles resume scenarios

    Args:
        dataset_key (str): Dataset being used
        device_info (dict): System and device information
        training_config (dict): Training configuration parameters
        experiment_name (str, optional): Custom experiment name
        resume_from_epoch (int, optional): Epoch number we're resuming from

    Returns:
        CompositeEnhancedMetricsLogger: Configured composite metrics logger instance
    """
    logger = CompositeEnhancedMetricsLogger(dataset_key, experiment_name, resume_from_epoch=resume_from_epoch)

    # Set system and training info
    logger.set_system_info(device_info)
    logger.set_training_config(training_config)

    return logger

# Enhanced analysis function for composite logs
def analyze_composite_training_metrics(experiment_log_path):
    """
    Enhanced analysis script for composite (multi-session) training logs

    Args:
        experiment_log_path (str): Path to the complete training log JSON file
    """
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    print("📊 COMPOSITE TRAINING METRICS ANALYSIS")
    print("=" * 60)

    try:
        # Load the complete training log
        with open(experiment_log_path, 'r') as f:
            data = json.load(f)

        print(f"✅ Loaded composite training data:")
        print(f"   🆔 Experiment: {data['experiment_info']['experiment_id']}")
        print(f"   📊 Total epochs: {data['experiment_info']['total_epochs_completed']}")
        print(f"   📈 Total steps: {data['experiment_info']['total_steps_completed']}")
        print(f"   🔄 Training sessions: {data['experiment_info']['total_training_sessions']}")

        # Display session information
        sessions = data.get('training_sessions', [])
        if sessions:
            print(f"\n🔄 Training Sessions:")
            for session in sessions:
                session_id = session['session_id']
                start_epoch = session.get('resume_from_epoch', 0)
                epochs_completed = session.get('epochs_completed', 'ongoing')
                print(f"   Session {session_id}: Started from epoch {start_epoch}, completed {epochs_completed} epochs")

        # Extract step metrics
        step_metrics = data['step_metrics']
        epochs = [step['epoch'] for step in step_metrics]
        d_losses = [step['losses']['discriminator_loss'] for step in step_metrics]
        g_losses = [step['losses']['generator_loss'] for step in step_metrics]
        grad_norms = [step['wgan_gp']['gradient_norm'] for step in step_metrics]
        health_scores = [step['health_indicators']['overall_health_score'] for step in step_metrics]
        session_ids = [step.get('session_id', 1) for step in step_metrics]

        # Create enhanced analysis plots
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Loss evolution with session boundaries
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(d_losses, label='Discriminator Loss', color='red', alpha=0.7)
        plt.plot(g_losses, label='Generator Loss', color='blue', alpha=0.7)

        # Add vertical lines for session boundaries
        current_session = session_ids[0] if session_ids else 1
        for i, session_id in enumerate(session_ids):
            if session_id != current_session:
                plt.axvline(x=i, color='green', linestyle='--', alpha=0.6, label=f'Session {session_id}' if session_id != current_session else "")
                current_session = session_id

        plt.title('Loss Evolution Across Sessions')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Gradient norms with session colors
        ax2 = plt.subplot(2, 3, 2)
        unique_sessions = list(set(session_ids))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sessions)))

        for session_id, color in zip(unique_sessions, colors):
            session_indices = [i for i, sid in enumerate(session_ids) if sid == session_id]
            session_grad_norms = [grad_norms[i] for i in session_indices]
            plt.scatter(session_indices, session_grad_norms,
                        label=f'Session {session_id}', color=color, alpha=0.6, s=10)

        plt.axhline(y=1.0, color='red', linestyle='--', label='Target (1.0)')
        plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=1.2, color='orange', linestyle='--', alpha=0.5)
        plt.title('Gradient Norms by Session')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Health scores across sessions
        ax3 = plt.subplot(2, 3, 3)
        for session_id, color in zip(unique_sessions, colors):
            session_indices = [i for i, sid in enumerate(session_ids) if sid == session_id]
            session_health = [health_scores[i] for i in session_indices]
            plt.plot(session_indices, session_health,
                     label=f'Session {session_id}', color=color, alpha=0.7)

        plt.title('Training Health Score by Session')
        plt.xlabel('Training Step')
        plt.ylabel('Health Score (0-100)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Epoch summaries across sessions
        ax4 = plt.subplot(2, 3, 4)
        epoch_summaries = data['epoch_summaries']
        if epoch_summaries:
            epoch_nums = [summary['epoch'] for summary in epoch_summaries]
            avg_d_losses = [summary['losses']['avg_discriminator_loss'] for summary in epoch_summaries]
            avg_g_losses = [summary['losses']['avg_generator_loss'] for summary in epoch_summaries]
            epoch_sessions = [summary.get('session_id', 1) for summary in epoch_summaries]

            for session_id, color in zip(unique_sessions, colors):
                session_epoch_nums = [epoch_nums[i] for i, sid in enumerate(epoch_sessions) if sid == session_id]
                session_d_losses = [avg_d_losses[i] for i, sid in enumerate(epoch_sessions) if sid == session_id]
                session_g_losses = [avg_g_losses[i] for i, sid in enumerate(epoch_sessions) if sid == session_id]

                if session_epoch_nums:
                    plt.plot(session_epoch_nums, session_d_losses, 'o-',
                             label=f'Session {session_id} D Loss', color=color, alpha=0.8)
                    plt.plot(session_epoch_nums, session_g_losses, 's--',
                             label=f'Session {session_id} G Loss', color=color, alpha=0.6)

            plt.title('Epoch Average Losses by Session')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot 5: Session comparison
        ax5 = plt.subplot(2, 3, 5)
        if len(unique_sessions) > 1:
            session_final_losses = []
            session_labels = []

            for session_id in unique_sessions:
                session_steps = [step for step in step_metrics if step.get('session_id') == session_id]
                if session_steps:
                    final_d_loss = session_steps[-1]['losses']['discriminator_loss']
                    final_g_loss = session_steps[-1]['losses']['generator_loss']
                    session_final_losses.append([final_d_loss, final_g_loss])
                    session_labels.append(f'Session {session_id}')

            if session_final_losses:
                session_final_losses = np.array(session_final_losses)
                x = np.arange(len(session_labels))
                width = 0.35

                plt.bar(x - width/2, session_final_losses[:, 0], width,
                        label='Final D Loss', alpha=0.8)
                plt.bar(x + width/2, session_final_losses[:, 1], width,
                        label='Final G Loss', alpha=0.8)

                plt.xlabel('Training Session')
                plt.ylabel('Final Loss')
                plt.title('Final Losses by Session')
                plt.xticks(x, session_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)

        # Plot 6: Training progression timeline
        ax6 = plt.subplot(2, 3, 6)
        # Show cumulative training progress
        cumulative_epochs = []
        cumulative_steps = []
        session_boundaries = []

        step_count = 0
        for i, step in enumerate(step_metrics):
            cumulative_steps.append(step_count)
            cumulative_epochs.append(step['epoch'])

            if i > 0 and step.get('session_id') != step_metrics[i-1].get('session_id'):
                session_boundaries.append(i)

            step_count += 1

        plt.plot(cumulative_steps, cumulative_epochs, 'b-', alpha=0.7, linewidth=2)

        for boundary in session_boundaries:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.8,
                        label='Session Boundary' if boundary == session_boundaries[0] else "")

        plt.title('Training Progression Timeline')
        plt.xlabel('Cumulative Steps')
        plt.ylabel('Epoch Number')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Composite Training Analysis: {data["experiment_info"]["experiment_id"]}', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Print enhanced summary statistics
        print(f"\n📈 COMPOSITE TRAINING SUMMARY:")
        print(f"   📊 Total training sessions: {len(unique_sessions)}")
        print(f"   📊 Final discriminator loss: {d_losses[-1]:.6f}")
        print(f"   📊 Final generator loss: {g_losses[-1]:.6f}")
        print(f"   📏 Final gradient norm: {grad_norms[-1]:.6f} (target: ~1.0)")
        print(f"   💯 Final health score: {health_scores[-1]:.1f}/100")
        print(f"   📈 Average health score: {np.mean(health_scores):.1f}/100")

        # Session-specific statistics
        for session_id in unique_sessions:
            session_steps = [step for step in step_metrics if step.get('session_id') == session_id]
            if session_steps:
                session_health = [step['health_indicators']['overall_health_score'] for step in session_steps]
                print(f"   🔄 Session {session_id}: {len(session_steps)} steps, avg health: {np.mean(session_health):.1f}/100")

        if epoch_summaries:
            final_stats = data['training_statistics']
            print(f"   🏆 Best Wasserstein distance: {final_stats['final_metrics']['best_wasserstein_distance']:.6f}")
            print(f"   ⏱️  Total training time: {final_stats['total_training_time_hours']:.2f} hours")
            print(f"   🔄 Total resumes: {final_stats['resume_summary']['total_resumes']}")

        print(f"\n✅ Composite analysis complete! Session boundaries and progression shown in plots.")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print(f"💡 Make sure the JSON file path is correct and contains composite training data")