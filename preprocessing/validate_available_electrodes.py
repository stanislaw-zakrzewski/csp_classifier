import numpy as np

from config_old import valid_channel_threshold


def validate_available_electrodes(subject, selected_channels, verbose=False):
    """Validator for selecting channels from Subject raw.

    Parameters
    ----------
    subject : Subject
        Subject class instance containing raw data that we want to select channels from.
    selected_channels : list of str
        List of channels that we intend to use that we want to check if are valid.
    verbose : boolean
        Enables/disables additional logging data. This might be useful to enable to see which channels
        were dropped due to lack of data or which are valid, but were not intended to be used.

    Returns
    -------
    valid_electrodes : list of str
        List of electrodes intended to be used in analysis that were also determined to contain valid data.
    """
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

    if verbose:
        print("ELECTRODE VALIDATION")
        print('Selected electrodes:')
        for key in selected_electrodes.keys():
            print(" - {}: {}".format(key, str(selected_electrodes[key])))
        print('Additional electrodes:')
        for key in additional_electrodes.keys():
            print(" - {}: {}".format(key, str(additional_electrodes[key])))

    return selected_electrodes['valid']
