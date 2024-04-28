def select_channels(signal, selected_channels_names):
    if len(selected_channels_names) > 0:
        for channel in signal.ch_names:
            if channel not in selected_channels_names:
                signal.drop_channels([channel])
