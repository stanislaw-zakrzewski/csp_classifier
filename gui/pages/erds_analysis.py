from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QScrollArea, QGridLayout)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from mne.io import read_raw_edf
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from config.config import Configurations
from data_classes.subject import Subject
from gui.fonts import fonts
from gui.colors import colors
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

class ERDSAnalysis(QScrollArea):
    def __init__(self, parent, controller, none=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.configurations = Configurations()
        
        self.available_electrodes = []
        self.picks_values = {}
        self.picks_buttons = {}
        self.left_canvases = []
        self.right_canvas = None
        self.selected_edf_file_path = ""
        
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        self.main_layout = QVBoxLayout(content_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        
        # Title (extracted component)
        app_title = TitleLabel("Kombajn EEG")
        self.main_layout.addWidget(app_title)
        
        # Back Button (extracted component)
        back_btn = BackButton(controller)
        self.main_layout.addWidget(back_btn)
        
        # Parameters section
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(params_widget)
        
        file_row = QHBoxLayout()
        params_layout.addLayout(file_row)
        
        self.select_btn = QPushButton("Select EDF file")
        self.select_btn.clicked.connect(self.select_edf_file)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
            }
        """)
        file_row.addWidget(self.select_btn)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 8px; color: #aaaaaa;")
        file_row.addWidget(self.file_label)
        file_row.addStretch()
        
        # Picks section
        picks_label = QLabel("Picks:")
        picks_label.setFont(fonts['medium_bold'])
        picks_label.setStyleSheet("color: white;")
        params_layout.addWidget(picks_label)
        
        self.picks_widget = QWidget()
        self.picks_grid = QGridLayout(self.picks_widget)
        self.picks_grid.setSpacing(5)
        params_layout.addWidget(self.picks_widget)
        
        # Analyze Button
        self.analyze_button = QPushButton("Analyze ERD/S for selected EDF")
        self.analyze_button.setFont(fonts['medium_font'])
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_edf_gui)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                padding: 10px; 
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)
        params_layout.addWidget(self.analyze_button)
        
        # Figures section layout container
        self.figures_widget = QWidget()
        self.figures_layout = QGridLayout(self.figures_widget)
        self.main_layout.addWidget(self.figures_widget)

    def select_edf_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select EDF file", "", "European Data Format files (*.edf)"
        )
        if filename:
            self.selected_edf_file_path = filename
            self.file_label.setText(filename)
            self.analyze_button.setEnabled(True)
            self.available_electrodes = read_raw_edf(filename, preload=False).info.ch_names
            
            # Clear old picks buttons
            self.picks_values = {}
            for name in self.picks_buttons:
                self.picks_buttons[name].deleteLater()
            self.picks_buttons = {}
            
            # Recreate picks buttons (8 per row)
            cols = 8
            for index, electrode_name in enumerate(self.available_electrodes):
                btn = QPushButton(electrode_name)
                btn.setStyleSheet("background-color: #1c1c1c; color: white; padding: 5px; border-radius: 3px;")
                btn.clicked.connect(lambda checked=False, el=electrode_name: self.toggle_pick_electrode(el))
                r = index // cols
                c = index % cols
                self.picks_grid.addWidget(btn, r, c)
                
                self.picks_buttons[electrode_name] = btn
                self.picks_values[electrode_name] = False
                
    def toggle_pick_electrode(self, electrode_name):
        self.picks_values[electrode_name] = not self.picks_values[electrode_name]
        btn = self.picks_buttons[electrode_name]
        if self.picks_values[electrode_name]:
            btn.setStyleSheet("background-color: red; color: white; padding: 5px; border-radius: 3px;")
        else:
            btn.setStyleSheet("background-color: #1c1c1c; color: white; padding: 5px; border-radius: 3px;")

    def car(self, data):
        averaged = np.sum(data, axis=0) / data.shape[0]
        for channel in data:
            channel -= averaged
        return data

    def analyze_edf_gui(self):
        if not self.selected_edf_file_path:
            return
            
        picks = [el for el in self.available_electrodes if self.picks_values[el]]
        if not picks:
            picks = [self.available_electrodes[0]]
            
        # Clear old canvases
        # Clear old canvases and close figures to avoid memory leaks
        for canvas in self.left_canvases:
            self.figures_layout.removeWidget(canvas)
            plt.close(canvas.figure)
            canvas.deleteLater()
        self.left_canvases = []
        if self.right_canvas is not None:
            self.figures_layout.removeWidget(self.right_canvas)
            plt.close(self.right_canvas.figure)
            self.right_canvas.deleteLater()
            self.right_canvas = None
            
        # Analysis
        subject = Subject(self.selected_edf_file_path)
        raw = subject.get_raw_copy()
        
        event_names = list(set(raw.annotations.description))
        event_names.sort()
        event_ids = {name: idx for idx, name in enumerate(event_names)}
        duration = raw.annotations.duration.max()
        sampling_frequency = int(raw.info['sfreq'])
        
        raw.filter(2, 36, l_trans_bandwidth=2, h_trans_bandwidth=2,
                   filter_length=sampling_frequency * 2,
                   fir_design='firwin',
                   skip_by_annotation='edge', verbose='ERROR')
        
        raw.rename_channels(lambda x: x.strip("."))
        
        tmin, tmax = -1, duration
        epochs = mne.Epochs(
            raw,
            events=subject.events,
            event_id=event_names,
            tmin=tmin - 0.5,
            tmax=tmax + 0.5,
            picks=picks,
            baseline=None,
            preload=True,
        )
        
        freqs = np.arange(2, 36)
        vmin, vmax = -1, 1.5
        baseline = (-1, 0)
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        kwargs = dict(
            n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
        )
        
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
        
        # Left canvases (dynamic per event type)
        for event_index, event in enumerate(event_ids):
            tfr_ev = tfr[event]
            ncols = len(picks) + 1
            ratios = [10 for _ in picks]
            ratios.append(1)
            fig, axes = plt.subplots(
                1, ncols, figsize=(12, ncols), gridspec_kw={"width_ratios": ratios}
            )
            
            # Apply dark mode to left figures
            fig.patch.set_facecolor('#121212')
            
            if ncols == 2:
                axes_list = [axes[0]]
                colorbar_ax = axes[1]
            else:
                axes_list = axes[:-1]
                colorbar_ax = axes[-1]

            for ch, ax in enumerate(axes_list):
                _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
                _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)
                c = np.stack(c1 + c2, axis=2)
                p = np.concatenate((p1, p2))
                mask = c[..., p <= 0.05].any(axis=-1)
                
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
                
                # Apply dark styles
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#2d2d2d')
                    
                ax.set_title(epochs.ch_names[ch], fontsize=10)
                ax.axvline(0, linewidth=1, color="black", linestyle=":")
                if ch != 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels("")
                    
            cbar = fig.colorbar(axes_list[0].images[-1], cax=colorbar_ax)
            cbar.ax.set_yscale("linear")
            cbar.ax.yaxis.label.set_color('white')
            cbar.ax.tick_params(colors='white')
            fig.suptitle(f"ERDS ({event})", color='white')
            
            canvas = FigureCanvas(fig)
            self.left_canvases.append(canvas)
            self.figures_layout.addWidget(canvas, event_index, 0)
            
        # Right canvas (general overview)
        df = tfr.to_data_frame(time_format=None, long_format=True)
        freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
        df["band"] = pd.cut(df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:])
        freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
        df = df[df.band.isin(freq_bands_of_interest)]
        df["band"] = df["band"].cat.remove_unused_categories()
        df["channel"] = df["channel"].cat.reorder_categories(picks, ordered=True)
        
        g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
        g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
        axline_kw = dict(color="white", linestyle="dashed", linewidth=0.5, alpha=0.5)
        g.map(plt.axhline, y=0, **axline_kw)
        g.map(plt.axvline, x=0, **axline_kw)
        g.set(ylim=(-1.5, 1.5))
        g.set_axis_labels("Time (s)", "ERDS")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.add_legend(ncol=2, loc="lower center")
        g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
        
        # Apply dark mode styles to FacetGrid
        g.figure.patch.set_facecolor('#121212')
        for ax in g.axes.flat:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for text in ax.texts:
                text.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#2d2d2d')
                
        # Fix label text colors specifically
        for ax in g.axes[:, -1]:
            ax.yaxis.get_label().set_color('white')
        for ax in g.axes[0, :]:
            ax.xaxis.get_label().set_color('white')
            
        legend = g.legend
        if legend:
            legend.get_frame().set_facecolor('#1e1e1e')
            legend.get_frame().set_edgecolor('#2d2d2d')
            for text in legend.get_texts():
                text.set_color('white')
        
        self.right_canvas = FigureCanvas(g.figure)
        self.figures_layout.addWidget(self.right_canvas, 0, 1, len(event_ids), 1)
