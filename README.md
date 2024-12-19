# EEG tools

## Parafac Analysis
Parafac analysis is contained in [parafac_analysis](parafac_analysis) folder.
It allows for detection of statistically significant parts of the EEG signal.
Please refer to the [docs](parafac_analysis/README.md) to get more information.

## ERD/ERS
Searching for Event Related Synchronizations (ERS) and Event Related Desynchronizations (ERD) is performed [ERD_ERS.py](ERD_ERS.py).

### Running
In order to run the ERD/ERS analysis on the single EDF file run `ERD_ERS.py` script.

### Parameters
During the run of ERD/ERS analysis a number of parameters will need to be passed:
* frequency band - only frequencies from this band will be processed and ultimately shown on heatmaps.
* time from cue (beginning of the task) to start - when we want the analysis to start, usually we want to start it just before cue to make sure we don't miss some activity
* time from cue (beginning of the task) to end - when we want the analysis to end, usually just after the task instructions disappear
* three selected channels - current version of the ERD/ERS analysis only supports three channels, default config is best for left\right motor imagery but will also work for any other hand motor imagery or movement tasks
* first label - the first task type (like left hand imagery movement) we want to perform the analysis for
* first label - the second task type (like right hand imagery movement)  we want to perform the analysis for

[//]: # (# CSP classifier)

[//]: # ()
[//]: # (## Setup)

[//]: # (Change paths in `config.py` file)

[//]: # ()
[//]: # (## Collecting data)

[//]: # (Before collecting data make sure that configuration is appropriate for device &#40;electrode configuration&#41; and subject &#40;name and gender&#41; in `config.py`file.)

[//]: # ()
[//]: # (Make sure that device is connected &#40;slow blinking on both receiver and transmitter&#41;, EEG cap is aligned, REF and GND electrodes are connected, anti-static bracelet is on the forearm and no electronic device is close to subject&#40;2m&#41;.)

[//]: # ()
[//]: # (Run `collect_data.py` script, results will be saved in `data` directory with timestamp&#40;acquisition finish&#41; in the filename.)

[//]: # ()
[//]: # (## Analyzing data)

[//]: # (Make sure that `subject_to_analyze` variable in `config.py` is set correctly.)

[//]: # ()
[//]: # (Run `analyze_data.py` script, it can take some time, depending on configurations that will be tested.)

[//]: # ()
[//]: # (## Browse EDF file)

[//]: # (You can see the raw &#40;lowpass to 40Hz&#41; EDF files using `visualize_edf_in_mne_browser` from `visualization/edf_in_mne_browser.py`.)

[//]: # ()
[//]: # (## Real time classification)

[//]: # (Classifier trained on the specified run &#40;`real_time_train_data`&#41; can be run in near real-time give feedback on detected activity to subject.)