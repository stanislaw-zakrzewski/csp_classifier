# CSP classifier

## Setup
Change paths in `config.py` file

## Collecting data
Before collecting data make sure that configuration is appropriate for device (electrode configuration) and subject (name and gender) in `config.py`file.

Make sure that device is connected (slow blinking on both receiver and transmitter), EEG cap is aligned, REF and GND electrodes are connected, anti-static bracelet is on the forearm and no electronic device is close to subject(2m).

Run `collect_data.py` script, results will be saved in `data` directory with timestamp(acquisition finish) in the filename.

## Analyzing data
Make sure that `subject_to_analyze` variable in `config.py` is set correctly.

Run `analyze_data.py` script, it can take some time, depending on configurations that will be tested.

## Browse EDF file
You can see the raw (lowpass to 40Hz) EDF files using `visualize_edf_in_mne_browser` from `visualization/edf_in_mne_browser.py`.