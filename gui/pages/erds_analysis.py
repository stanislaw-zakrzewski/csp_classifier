from tkinter import *
from tkinter import filedialog as fd

import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from analyze_data import analyze_edf as analyze_edf_prime
from config.config import Configurations
from gui.colors import colors
from gui.components.double_scrolled_frame import DoubleScrolledFrame
from gui.fonts import fonts
from gui.pages.start_page import StartPage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from data_classes.subject import Subject
from classifiers.parafac import process as parafacProcess
from classifiers.flat import process as cspProcess


class ERDSAnalysis(DoubleScrolledFrame):
    def __init__(self, parent, controller):
        DoubleScrolledFrame.__init__(self, parent)
        self.configurations = Configurations()

        app_title = Label(self, text="Kombajn EEG", font=fonts['large_bold_font'])
        app_title.grid(row=0, column=0, padx=10, pady=10, columnspan=10, sticky='W')

        back_to_start_page_button = Button(self, text="Back to Start Page",
                                           command=lambda: controller.show_frame(StartPage))
        back_to_start_page_button.grid(row=1, column=0, padx=10, pady=10)

        Button(self, text='Select EDF file', command=self.select_edf_file).grid(row=2, column=0, padx=10, pady=10)
        self.selected_edf_file = StringVar()
        self.selected_edf_file.set('')
        Label(self, textvariable=self.selected_edf_file).grid(row=2, column=1)
        Button(self, text='Analyze selected EDF', command=self.analyze_edf_gui).grid(row=3, column=0, padx=10, pady=10)
        self.right_canvas = None
        self.left_canvases = []

    def select_edf_file(self):
        filename = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
        if filename:
            self.selected_edf_file.set(filename)

    def analyze_edf_gui(self):
        if self.selected_edf_file.get() != '':
            PICKS = ('C3', 'CZ', 'C4')
            ANNOTATIONS_RENAME_DICT = dict(rest="rest", movement="movement")
            EVENT_IDS = dict(movement=2, rest=3)
            EVENT_NAMES = ['rest', 'movement']
            DURATION = 11

            subject = Subject(self.selected_edf_file.get())

            raw = subject.get_raw_copy()
            sampling_frequency = int(raw.info['sfreq'])

            raw.filter(2, 36, l_trans_bandwidth=2, h_trans_bandwidth=2,
                       filter_length=sampling_frequency * 2,
                       fir_design='firwin',
                       skip_by_annotation='edge', verbose='ERROR')

            raw.rename_channels(lambda x: x.strip("."))  # remove dots from channel names
            # rename descriptions to be more easily interpretable
            raw.annotations.rename(ANNOTATIONS_RENAME_DICT)

            tmin, tmax = -1, DURATION
            event_ids = EVENT_IDS  # map event IDs to tasks

            epochs = mne.Epochs(
                raw,
                event_id=EVENT_NAMES,
                tmin=tmin - 0.5,
                tmax=tmax + 0.5,
                picks=PICKS,
                baseline=None,
                preload=True,
            )

            freqs = np.arange(2, 36)  # frequencies from 2-35Hz
            vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
            baseline = (-1, 0)  # baseline interval (in s)
            cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

            kwargs = dict(
                n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
            )  # for cluster test

            tfr = epochs.compute_tfr(
                method="multitaper",
                freqs=freqs,
                n_cycles=freqs,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=2,
            )
            tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")

            for event_index, event in enumerate(event_ids):
                # select desired epochs for visualization
                tfr_ev = tfr[event]
                fig, axes = plt.subplots(
                    1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
                )
                for ch, ax in enumerate(axes[:-1]):  # for each channel
                    # positive clusters
                    _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
                    # negative clusters
                    _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

                    # note that we keep clusters with p <= 0.05 from the combined clusters
                    # of two independent tests; in this example, we do not correct for
                    # these two comparisons
                    c = np.stack(c1 + c2, axis=2)  # combined clusters
                    p = np.concatenate((p1, p2))  # combined p-values
                    mask = c[..., p <= 0.05].any(axis=-1)

                    # plot TFR (ERDS map with masking)
                    tfr_ev.average().plot(
                        [ch],
                        cmap="RdBu",
                        cnorm=cnorm,
                        axes=ax,
                        colorbar=False,
                        show=False,
                        mask=mask,
                        mask_style="mask",
                    )

                    ax.set_title(epochs.ch_names[ch], fontsize=10)
                    ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
                    if ch != 0:
                        ax.set_ylabel("")
                        ax.set_yticklabels("")
                fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
                fig.suptitle(f"ERDS ({event})")

                if len(self.left_canvases) <= event_index:
                    self.left_canvases.append(FigureCanvasTkAgg(fig, master=self))  # A tk.DrawingArea.
                else:
                    self.left_canvases[event_index].get_tk_widget().destroy()
                    self.left_canvases[event_index] = FigureCanvasTkAgg(fig, master=self)
                self.left_canvases[event_index].draw()
                self.left_canvases[event_index].get_tk_widget().grid(row=4 + event_index, column=0)

            df = tfr.to_data_frame(time_format=None)
            df.head()

            df = tfr.to_data_frame(time_format=None, long_format=True)

            # Map to frequency bands:
            freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
            df["band"] = pd.cut(
                df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
            )

            # Filter to retain only relevant frequency bands:
            freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
            df = df[df.band.isin(freq_bands_of_interest)]
            df["band"] = df["band"].cat.remove_unused_categories()

            # Order channels for plotting:
            df["channel"] = df["channel"].cat.reorder_categories(PICKS, ordered=True)

            g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
            g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
            axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
            g.map(plt.axhline, y=0, **axline_kw)
            g.map(plt.axvline, x=0, **axline_kw)
            g.set(ylim=(None, 1.5))
            g.set_axis_labels("Time (s)", "ERDS")
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.add_legend(ncol=2, loc="lower center")
            g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)

            if self.right_canvas is None:
                self.right_canvas = FigureCanvasTkAgg(g.figure, master=self)  # A tk.DrawingArea.
            else:
                self.right_canvas.get_tk_widget().destroy()
                self.right_canvas = FigureCanvasTkAgg(g.figure, master=self)
            self.right_canvas.draw()
            self.right_canvas.get_tk_widget().grid(row=4, column=1, rowspan=10)


            # # accuracy_data = analyze_edf_prime(self.selected_edf_file.get(),
            # #                             classifier_type=self.configurations.read('analyze_data.classifier'),
            # #                             verbose='ERROR')
            # # figure = Figure(figsize=(25, 10))
            # # ax = figure.subplots()
            # # accuracy_data.to_csv('stacked_mlp_space.csv')  # TODO remove this
            # sns.lineplot(data=accuracy_data, x="frequency", y="accuracy", hue="configuration", errorbar=None, ax=ax, markers=True, style='configuration')
            #
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
            # ax.grid()
            # if self.canvas is None:
            #     self.canvas = FigureCanvasTkAgg(figure, master=self)  # A tk.DrawingArea.
            # else:
            #     self.canvas.get_tk_widget().destroy()
            #     self.canvas = FigureCanvasTkAgg(figure, master=self)
            # self.canvas.draw()
            # self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=10)
            # toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            # toolbar.update()
            # toolbar.grid(row=5, column=0, columnspan=10)
