# PARAFAC Analysis
PARAFAC analysis is used to find statistically significant parts of the EEG signal related to the task performed by the subject captured in the recording.

## Setup
Make sure that `significant_heatmaps` folder exists in the `parafac_analysis` directory. Otherwise results of the analysis are not going to be saved.

## Parameters
During the runtime the scripts will ask for parameters. Some of the terms needs to be explained:
* `rank` - PARAFAC decomposition parameter, which determines how many atoms will be created during the decomposition. During the runtime user will be asked to provide `start rank` and `end rank`. Those terms refer to the minimum and maximum rank parameter values for the optimizer. Selecting `start rank` below 2 is pointless and selecting `end rank` too high will result in absurd runtimes. I'd suggest starting with 5-25 range.
* `replica count` - PARAFAC decomposition can at random result in a very unfortunate decomposition, so being able to run it more than once for a given rank is crucial. This parameter does exactly that, determines how many times the decomposition will be run for the given rank. Default value of 5 should be enough.
* `optimizer iterations` - PARAFAC analysis finds the best rank for decomposition using Bayesian optimizer. This parameter determines how many times this optimizer runs. Setting it to more that the cardinality of ranks from `start rank` to `end rank` range is pointless. Around 50% of the range length should be enough.
* `montage` - EEG cap electrode setup that was used during recording, used for spatial laplacian filtering CSD (Current Source Density).
* `cue` - point in time in which the instruction for a given task appeared. This is used to "cut" the trial. Usually we select start of the cut to .5 seconds after cue and end to the end of cue (so we just pass length of the cue).
* `label` - the name of the task.


## Running PARAFAC analysis
It is advised to run the single file first to make sure the parameters are tuned correctly.

### Processing single file
In order to run the PARAFAC analysis on the single EDF file run `processing/parafac_analysis_file.py` script.
For pycharm remember to change working directory to the main repo folder. 

### Processing multiple files
In order to run the PARAFAC analysis on multiple EDF files run `processing/parafac_analysis_folder.py` script.
For pycharm remember to change working directory to the main repo folder.

All EDF files in the folder should ideally be similar in trial type, trial count etc. as parameters for the PARAFAC analysis run are shared between all the files.

## Results of the analysis
Results are saved in the 'significant_heatmaps' folder in numpy using Pickle. The structure is as follows:
* `heatmap` - heatmap of significant atoms for both classes  
* `heatmap_a` - heatmap of significant atoms for class `a`
* `heatmap_b` - heatmap of significant atoms for class `b`
* `frequencies` - list of frequencies corresponding to the second dimension of the heatmaps
* `channels` - list of channels corresponding to the first dimension of the heatmaps
* `starting_rank` - min rank for the PARAFAC decomposition
* `end_rank` - max rank for the PARAFAC decomposition
* `replica_count` - how many times PARAFAC decomposition for each rank will be performed
* `optimizer_iterations` - how many times optimizer will search for the best rank
* `a_label_name` - name of the "first" `a` class (task name)
* `b_label_name` - name of the "second" `b` class (task name)
* `t_min` - start of the trial (related to start of the displayed instruction, cue)
* `t_max` - end of the trial (related to start of the displayed instruction, cue)
* `montage` - which montage was used to record the EEG signal