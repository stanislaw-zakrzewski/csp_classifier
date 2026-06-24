import time
import numpy as np
from datetime import datetime

from config.config import Configurations

class ZeroMockBCIInterface:
    def __init__(self):
        self.configurations = Configurations()
        self.sampling_rate = self.configurations.read("general.sampling_rate")
        
    def run_acquisition(self, prompt_viewer, current_queue, update_experiment_timeline_plot, progressbar_value, batches_per_second, real_time_processor=None):
        batch_size = self.sampling_rate // batches_per_second
        
        signal = [[] for _ in range(32)]
        start_date = datetime.now()
        
        while current_queue is not None and len(current_queue) > 0:
            # Append zeros to all 32 channels
            for channel in range(32):
                signal[channel] = np.concatenate((signal[channel], list(np.zeros(batch_size))))
                
            item = current_queue[0]
            prompt_to_show = item[0]
            
            if real_time_processor is not None:
                prompt_text = real_time_processor(signal, current_true_label=prompt_to_show)
                if prompt_text is not None:
                    prompt_to_show = prompt_text
            
            print('ZERO MOCK BCI', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], len(signal[0]), prompt_to_show)
            
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
