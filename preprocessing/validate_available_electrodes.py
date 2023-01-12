import numpy as np

from config import valid_channel_threshold


def validate_available_electrodes(subject, selected_channels):
    raw = subject.get_raw_copy()
    raw.filter(l_freq=2, h_freq=40, verbose='ERROR')

    selected_electrodes = {'valid': [], 'invalid': [], 'not found': []}
    additional_electrodes = {'valid': [], 'invalid': []}

    selected_electrodes['not found'] = list(set(selected_channels) - set(subject.electrode_names))

    for electrode_name in subject.electrode_names:
        average_channel_power = np.average(np.absolute(raw.get_data(picks=[electrode_name])))
        if average_channel_power > valid_channel_threshold:
            if electrode_name in selected_channels:
                selected_electrodes['valid'].append(electrode_name)
            else:
                additional_electrodes['valid'].append(electrode_name)
        else:
            if electrode_name in selected_channels:
                selected_electrodes['invalid'].append(electrode_name)
            else:
                additional_electrodes['invalid'].append(electrode_name)

    print("ELECTRODE VALIDATION")
    print('Selected electrodes:')
    for key in selected_electrodes.keys():
        print(" - {}: {}".format(key, str(selected_electrodes[key])))
    print('Additional electrodes:')
    for key in additional_electrodes.keys():
        print(" - {}: {}".format(key, str(additional_electrodes[key])))

    return selected_electrodes['valid']
