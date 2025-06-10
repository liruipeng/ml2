import os
import json
import numpy as np
import datetime # Import datetime for timestamping logs

class ExperimentLogger:
    def __init__(self, log_dir, rank_val):
        self.log_dir = log_dir
        self.rank_val = rank_val
        # Each rank will have its own log file and plot data directory
        self.run_log_file = os.path.join(self.log_dir, f'log_run{self.rank_val+1}.txt')
        self.plot_data_dir = os.path.join(self.log_dir, f'plots_run{self.rank_val+1}')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_data_dir, exist_ok=True)

        # Initialize the log file with a header for this specific rank's log
        with open(self.run_log_file, 'w') as f:
            f.write(f"--- Training Log for Rank {self.rank_val} ---\n")
            f.write(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")

    def log_message(self, message):
        """
        Logs a message to this rank's specific training log file.
        """
        with open(self.run_log_file, 'a') as f:
            f.write(f"{message}\n")

    def log_scalar(self, tag, value, step):
        """
        Logs a scalar value to this rank's specific training log file.
        This replaces the TensorBoard scalar logging with text output.
        """
        # CSV format for easier parsing
        log_entry = f"Step {step}: {tag} = {value:.6e}"
        self.log_message(log_entry)

    def save_plot_data(self, data_dict, filename_prefix, step):
        """
        Saves plot data to a .npz file within this rank's plot_data directory.
        """
        filepath = os.path.join(self.plot_data_dir, f"{filename_prefix}_epoch_{step:08d}.npz")
        np.savez_compressed(filepath, **data_dict)
        self.log_message(f"Saved plot data to {os.path.basename(filepath)}")

    def close(self):
        """
        No specific closing actions needed for simple file logging.
        Retained for API consistency.
        """
        self.log_message(f"Log for Rank {self.rank_val} closed.")

    def save_config(self, config_dict):
        """
        Saves the experiment configuration to a JSON file in this rank's log directory.
        This ensures each run's config is saved with its logs.
        """
        config_filepath = os.path.join(self.log_dir, 'config.json')
        with open(config_filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        self.log_message(f"Configuration saved to {os.path.basename(config_filepath)}")
