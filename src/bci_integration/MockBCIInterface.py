import mne
import time
import numpy as np
from datetime import datetime

from config.config import Configurations
from src.bci_integration.GtecNautilusProInterface import channel_index_to_name

class MockBCIInterface:
    def __init__(self, edf_path):
        self.edf_path = edf_path
        self.configurations = Configurations()
        self.selected_channels = self.configurations.read('general.selected_electrodes')
        
    def run_acquisition(self, prompt_viewer, current_queue, update_experiment_timeline_plot, progressbar_value, batches_per_second, real_time_processor=None):
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        data = raw.get_data() # shape: (channels, times)
        sfreq = int(raw.info['sfreq'])
        channel_names = raw.ch_names
        
        batch_size = sfreq // batches_per_second
        total_samples = data.shape[1]
        
        # Determine existing channels mapping to match GtecNautilusProInterface exactly
        existing_channels = {}
        valid_channel_count = 0
        
        # EDF might have 32 channels. We map them by name.
        name_to_edf_idx = {name: idx for idx, name in enumerate(channel_names)}
        
        for ch_idx in range(32):
            name = channel_index_to_name.get(ch_idx)
            if name in self.selected_channels and name in name_to_edf_idx:
                existing_channels[ch_idx] = name_to_edf_idx[name]
            else:
                existing_channels[ch_idx] = -1
                
        signal = [[] for _ in range(32)]
        start_date = datetime.now()
        
        for start_idx in range(0, total_samples, batch_size):
            if current_queue is None or len(current_queue) == 0:
                break
                
            end_idx = min(start_idx + batch_size, total_samples)
            if start_idx == end_idx:
                break
                
            chunk = data[:, start_idx:end_idx] # shape: (channels, batch_size)
            
            # Mimic the processCallback logic
            for channel in range(32):
                edf_index = existing_channels[channel]
                if edf_index != -1:
                    signal[channel] = np.concatenate((signal[channel], list(chunk[edf_index, :])))
                else:
                    signal[channel] = np.concatenate((signal[channel], list(np.zeros(end_idx - start_idx))))
                    
            item = current_queue[0]
            
            prompt_to_show = item[0]
            if real_time_processor is not None:
                prompt_text = real_time_processor(signal, current_true_label=prompt_to_show)
                if prompt_text is not None:
                    prompt_to_show = prompt_text
                    
            print('MOCK BCI', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], len(signal[0]), prompt_to_show)
            
            if not prompt_viewer.closed:
                prompt_viewer.change_prompt(prompt_to_show)
                
                # Decrement duration
                item[1] -= 1.0 / batches_per_second
                item[1] = round(item[1], 1)
                
                if item[1] < 0.1 and current_queue is not None:
                    current_queue.pop(0)
                    
                update_experiment_timeline_plot(1.0 / batches_per_second)
            else:
                break
                
            # Simulate real-time delay
            time.sleep(1.0 / batches_per_second)
            
        return signal, start_date
