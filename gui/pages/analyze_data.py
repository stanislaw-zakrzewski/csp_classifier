import sys
import os
import tempfile
import shutil
import traceback
import mne
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea
from PySide6.QtCore import Qt, QThread, Signal, QObject

from config.config import Configurations
from gui.colors import colors
from gui.fonts import fonts
from gui.components.back_button import BackButton
from gui.components.title_label import TitleLabel

# MOABB and sklearn / pyriemann
from moabb.datasets.base import BaseDataset
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

# Monkey-patch WithinSessionEvaluation to respect n_splits
def custom_create_splitter(self):
    from sklearn.model_selection import StratifiedKFold
    from moabb.evaluations.splitters import WithinSessionSplitter
    cv_class, cv_kwargs = self._resolve_cv(StratifiedKFold)
    n_folds = self.n_splits if self.n_splits is not None else 5
    return WithinSessionSplitter(
        n_folds=n_folds,
        shuffle=True,
        random_state=self.random_state,
        cv_class=cv_class,
        **cv_kwargs,
    )

WithinSessionEvaluation._create_splitter = custom_create_splitter

# Keep track of active QThreads globally to prevent them from being garbage-collected
# during execution, which would cause the fatal QThread destruction crash.
active_threads = []


class LocalEDF(BaseDataset):
    def __init__(self, edf_path):
        self.edf_path = edf_path
        
        # Load Raw file to inspect annotations
        raw = mne.io.read_raw_edf(edf_path, preload=False)
        
        # Get unique annotations (ignore break/pause/rest_break)
        unique_annots = sorted(list(set(raw.annotations.description)))
        unique_annots = [a for a in unique_annots if a not in ('break', 'pause', 'rest_break')]
        
        # Map unique annotations to numbers (1, 2, 3...)
        events = {desc: idx + 1 for idx, desc in enumerate(unique_annots)}
        
        # Set max duration from annotations
        max_duration = float(raw.annotations.duration.max()) if len(raw.annotations.duration) > 0 else 4.0
        interval = [0.0, max_duration]
        
        super().__init__(
            subjects=[1],
            sessions_per_subject=1,
            events=events,
            code="LocalEDF",
            interval=interval,
            paradigm="imagery"
        )
        self._raw_cache = mne.io.read_raw_edf(edf_path, preload=True)

    def _get_single_subject_data(self, subject):
        return {'0session': {'0run': self._raw_cache}}

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        return [self.edf_path]


class AnalysisWorker(QObject):
    analysis_finished = Signal(object, str, float)  # (DataFrame or None, error_message, chance_level)
    status = Signal(str)

    def __init__(self, edf_path):
        super().__init__()
        self.edf_path = edf_path

    def run(self):
        temp_dir = tempfile.mkdtemp(prefix="bids_temp_")
        try:
            self.status.emit("Reading EDF file annotations...")
            dataset = LocalEDF(self.edf_path)
            
            events_list = list(dataset.event_id.keys())
            if not events_list:
                raise ValueError("No class events found in EDF file annotations. Cannot perform classification.")
                
            n_classes = len(events_list)
            chance_level = 1.0 / n_classes
            
            # Calculate sample count from annotations
            annots = dataset._raw_cache.annotations
            n_samples = len([a for a in annots if a['description'] in events_list])
            if n_samples < 2:
                raise ValueError(f"Too few trials ({n_samples}) in EDF file. At least 2 trials are required for classification.")
                
            n_splits = min(5, n_samples)
            
            self.status.emit("Converting EDF file to BIDS format...")
            # Run BIDS conversion
            bids_root = dataset.convert_to_bids(path=temp_dir, subjects=[1], overwrite=True)
            
            self.status.emit(f"Running MOABB cross-validation ({n_splits}-fold)...")
            
            # Setup pipelines
            from pyriemann.tangentspace import TangentSpace
            from pyriemann.classification import MDM
            from conv_s4d import ConvS4DClassifier
            
            # The custom paradigm that uses the dynamic config
            from moabb.paradigms import MotorImagery
            
            pipelines = {
                "Riemannian MDM": make_pipeline(Covariances(estimator='lwf'), MDM()),
                "Cov + TS+ SVM": make_pipeline(Covariances(estimator='lwf'), TangentSpace(), SVC(kernel='linear')),
                "CSP + SVM": make_pipeline(CSP(n_components=4), SVC(kernel='rbf')),
                # "Conv-S4D (CNN)": ConvS4DClassifier()
            }
            
            paradigm = MotorImagery(events=events_list, n_classes=n_classes, fmin=2, fmax=36)
            
            from moabb.datasets.base import CacheConfig
            cache_config = CacheConfig(use=False, save_raw=False, save_epochs=False, save_array=False)
            
            evaluation = WithinSessionEvaluation(
                paradigm=paradigm,
                datasets=[dataset],
                overwrite=True,
                n_splits=n_splits,
                cv_class=StratifiedKFold,
                hdf5_path=os.path.join(temp_dir, "results.hdf5"),
                cache_config=cache_config,
                n_jobs=1
            )
            
            from joblib import parallel_backend
            with parallel_backend('sequential'):
                results = evaluation.process(pipelines)
                
            # Safely cleanup PyTorch objects in the worker thread before it exits
            del pipelines
            del evaluation
            import gc
            gc.collect()
            
            self.analysis_finished.emit(results, "", chance_level)
            
        except Exception as e:
            print("[AnalysisThread] CRITICAL: Exception occurred inside thread run:")
            traceback.print_exc()
            sys.stdout.flush()
            tb = traceback.format_exc()
            self.analysis_finished.emit(None, f"{str(e)}\n\n{tb}", 0.5)
        finally:
            print('Avoid errors')
            # Clean up the temp BIDS directory
            # Commented out to prevent memory-mapping crash on Windows
            # try:
            #     shutil.rmtree(temp_dir, ignore_errors=True)
            # except Exception:
            #     pass


class AnalyzeData(QScrollArea):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setStyleSheet(f"background-color: {colors['white_smoke']}; border: none;")
        self.configurations = Configurations()
        self.canvas = None
        self.toolbar = None
        self.selected_file_path = ""
        self.is_analyzing = False
        
        content_widget = QWidget()
        self.setWidget(content_widget)
        
        self.layout = QVBoxLayout(content_widget)
        self.layout.setAlignment(Qt.AlignTop)
        
        # App Title
        app_title = TitleLabel("Kombajn EEG")
        self.layout.addWidget(app_title)
        
        # Back button
        back_btn = BackButton(controller)
        self.layout.addWidget(back_btn)
        
        # File selector row
        file_row = QHBoxLayout()
        self.layout.addLayout(file_row)
        
        self.select_btn = QPushButton("Select EDF file")
        self.select_btn.clicked.connect(self.select_edf_file)
        self.select_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
            }}
            QPushButton:disabled {{
                background-color: #1a1a1a;
                color: #555555;
                border: 1px solid #222222;
            }}
        """)
        file_row.addWidget(self.select_btn)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 8px; color: #aaaaaa;")
        file_row.addWidget(self.file_label)
        file_row.addStretch()
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze selected EDF")
        self.analyze_btn.clicked.connect(self.analyze_edf_gui)
        self.analyze_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; 
                background-color: {colors['success']}; 
                color: white; 
                border: none; 
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['success_hover']};
            }}
            QPushButton:disabled {{
                background-color: #2e4a30;
                color: #888888;
            }}
        """)
        self.layout.addWidget(self.analyze_btn)
        self.analyze_btn.setEnabled(False)
        
        self.status_label = QLabel("")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setStyleSheet("padding: 8px; color: #2196F3; font-weight: bold;")
        self.layout.addWidget(self.status_label)
        
        # Post-analysis control panel (hidden by default)
        self.post_analysis_widget = QWidget()
        self.post_analysis_layout = QVBoxLayout(self.post_analysis_widget)
        self.post_analysis_layout.setContentsMargins(0, 10, 0, 10)
        self.layout.addWidget(self.post_analysis_widget)
        self.post_analysis_widget.hide()
        
        # Matplotlib chart container
        self.chart_container = QVBoxLayout()
        self.layout.addLayout(self.chart_container)

    def select_edf_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select EDF file", "", "European Data Format files (*.edf)"
        )
        if filename:
            self.selected_file_path = filename
            self.file_label.setText(filename)
            self.status_label.setText("")
            self.analyze_btn.setEnabled(True)
            
    def analyze_edf_gui(self):
        if not self.selected_file_path:
            self.status_label.setStyleSheet("color: #ff9800; font-weight: bold; padding: 4px;")
            self.status_label.setText("Please select an EDF file first.")
            return

        # Prevent concurrent executions / multiple click spamming
        if self.is_analyzing:
            return

        if hasattr(self, 'analysis_thread') and self.analysis_thread is not None:
            try:
                if self.analysis_thread.isRunning():
                    return
                # Ensure previous thread is fully joined
                self.analysis_thread.wait()
            except RuntimeError:
                # The C++ object was already deleted by deleteLater, so it is safe to proceed
                self.analysis_thread = None

        self.is_analyzing = True

        # Disable buttons
        self.select_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.status_label.setStyleSheet("color: #2196F3; font-weight: bold; padding: 4px;")
        self.status_label.setText("Starting analysis...")

        # Start thread
        self.worker_thread = QThread()
        self.worker = AnalysisWorker(self.selected_file_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        
        self.worker.analysis_finished.connect(self.analysis_completed)
        self.worker.status.connect(self.update_status)
        
        self.worker_thread.start()

    def update_status(self, text):
        self.status_label.setText(text)

    def analysis_completed(self, results, error_msg, chance_level):
        print(f"[AnalyzeData] analysis_completed called. error_msg: {error_msg}")
        sys.stdout.flush()
        
        self.is_analyzing = False
        self.select_btn.setEnabled(True)
        # Keep analyze button disabled until results are cleared
        self.analyze_btn.setEnabled(False)

        if error_msg:
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 4px;")
            self.status_label.setText(f"Analysis failed:\n{error_msg}")
            print(f"[AnalyzeData] Analysis failed: {error_msg}")
            sys.stdout.flush()
            return

        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 4px;")
        self.status_label.setText("Analysis completed successfully!")
        print("[AnalyzeData] Analysis completed successfully. Starting plotting...")
        sys.stdout.flush()

        # Plot results
        try:
            # Create a Figure with dark styling
            figure = Figure(figsize=(8, 5), facecolor='#121212')
            ax = figure.subplots()
            ax.set_facecolor('#1e1e1e')
            
            # Style the labels, ticks, and spine borders
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.grid(True, color='#2d2d2d', linestyle='--', alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color('#2d2d2d')
                
            # Plot using seaborn barplot
            print("[AnalyzeData] Generating Seaborn barplot...")
            sys.stdout.flush()
            sns.barplot(
                data=results,
                x="pipeline",
                y="score",
                hue="pipeline",
                legend=False,
                ax=ax,
                palette="viridis",
                errorbar="sd",
                capsize=0.1
            )
            
            # Draw chance level
            ax.axhline(y=chance_level, color='#f44336', linestyle='--', linewidth=1.5, label=f'Chance Level ({chance_level:.2f})')
            
            ax.set_xlabel("Classification Pipeline", fontsize=10, fontweight='bold', labelpad=10)
            ax.set_ylabel("Accuracy", fontsize=10, fontweight='bold', labelpad=10)
            ax.set_title("BCI Pipeline Comparison (MOABB Within-Session)", fontsize=12, fontweight='bold', pad=15)
            ax.set_ylim(0, 1.05)
            
            # Style the Legend for dark mode
            legend = ax.legend(facecolor='#1e1e1e', edgecolor='#2d2d2d')
            if legend:
                for text in legend.get_texts():
                    text.set_color('white')
                    
            # Remove old canvas and toolbar if any
            print("[AnalyzeData] Removing old canvas and toolbar if present...")
            sys.stdout.flush()
            if self.canvas is not None:
                self.chart_container.removeWidget(self.canvas)
                try:
                    self.canvas.figure.clf()
                except Exception as ce:
                    print(f"[AnalyzeData] Warning clearing figure: {ce}")
                self.canvas.deleteLater()
                self.canvas = None
            if self.toolbar is not None:
                self.chart_container.removeWidget(self.toolbar)
                self.toolbar.deleteLater()
                self.toolbar = None
                
            # Create and add new canvas & toolbar
            print("[AnalyzeData] Creating new FigureCanvas...")
            sys.stdout.flush()
            self.canvas = FigureCanvas(figure)
            # self.canvas.setMinimumHeight(400)
            self.toolbar = NavigationToolbar(self.canvas, self)
            
            # figure.tight_layout()
            # self.canvas.draw()
            
            self.chart_container.addWidget(self.canvas)
            self.chart_container.addWidget(self.toolbar)
            print("[AnalyzeData] Plot successfully added to GUI.")
            sys.stdout.flush()
            
        except Exception as e:
            print("[AnalyzeData] CRITICAL: Exception occurred while plotting results:")
            traceback.print_exc()
            sys.stdout.flush()
            tb = traceback.format_exc()
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 4px;")
            self.status_label.setText(f"Error plotting results: {e}\n\n{tb}")

        # Setup the post-analysis control panel buttons (always try to show this)
        try:
            print("[AnalyzeData] Setting up post-analysis buttons...")
            sys.stdout.flush()
            while self.post_analysis_layout.count():
                child = self.post_analysis_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                    
            # 1. Notice Label (instead of Clear button)
            notice_label = QLabel("Due to implementation of analysis the program must be reopened to run analysis again")
            notice_label.setStyleSheet("color: #ff9800; font-weight: bold; padding: 4px; font-size: 13px;")
            self.post_analysis_layout.addWidget(notice_label)
            
            # Spacer
            self.post_analysis_layout.addSpacing(10)
            
            # 2. Save buttons horizontal row (using exact Viridis colors)
            save_row_widget = QWidget()
            save_row_layout = QHBoxLayout(save_row_widget)
            save_row_layout.setContentsMargins(0, 5, 0, 5)
            
            pipelines_styles = {
                "Riemannian MDM": ("#440154", "#482878"),
                "Cov + TS+ SVM": ("#31688e", "#2c728e"),
                "CSP + SVM": ("#35b779", "#20a486"),
                # "Conv-S4D (CNN)": ("#fde725", "#d6c21a")
            }
            
            for name, (bg_color, hover_color) in pipelines_styles.items():
                btn = QPushButton(f"Save {name}")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        padding: 8px 15px; 
                        background-color: {bg_color}; 
                        color: white; 
                        border: none; 
                        border-radius: 4px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {hover_color};
                    }}
                """)
                btn.clicked.connect(lambda checked=False, p_name=name: self.save_pipeline(p_name))
                save_row_layout.addWidget(btn)
                
            save_row_layout.addStretch()
            self.post_analysis_layout.addWidget(save_row_widget)
            
            self.post_analysis_widget.show()
            print("[AnalyzeData] Post-analysis widget shown successfully.")
            sys.stdout.flush()
        except Exception as e:
            print("[AnalyzeData] CRITICAL: Exception occurred while setting up buttons:")
            traceback.print_exc()
            sys.stdout.flush()

    def save_pipeline(self, pipeline_name):
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save Trained {pipeline_name}", f"{pipeline_name.replace(' + ', '_').replace(' ', '_').lower()}.pkl", "Pickle files (*.pkl)"
        )
        if not filename:
            return
            
        try:
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold; padding: 4px;")
            self.status_label.setText(f"Fitting and saving {pipeline_name}...")
            
            # Re-create dataset and paradigm to fit on all data
            dataset = LocalEDF(self.selected_file_path)
            events_list = list(dataset.event_id.keys())
            
            # Setup specific pipeline structure
            if pipeline_name == "CSP + LDA":
                pipeline = make_pipeline(CSP(n_components=4), LDA())
            elif pipeline_name == "Cov + Tangent Space + LR":
                pipeline = make_pipeline(Covariances(estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(max_iter=1000))
            elif pipeline_name == "CSP + SVM":
                pipeline = make_pipeline(CSP(n_components=4), SVC(kernel='rbf'))
            elif pipeline_name == "Conv-S4D (CNN)":
                from conv_s4d import ConvS4DClassifier
                pipeline = ConvS4DClassifier()
            else:
                raise ValueError(f"Unknown pipeline: {pipeline_name}")
                
            paradigm = MotorImagery(events=events_list, n_classes=len(events_list), fmin=2, fmax=36)
            
            # Fit on all epochs of the dataset
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
            pipeline.fit(X, y)
            
            # Serialize the trained pipeline using pickle
            with open(filename, 'wb') as f:
                pickle.dump(pipeline, f)
                
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 4px;")
            self.status_label.setText(f"Successfully saved {pipeline_name} to {os.path.basename(filename)}!")
        except Exception as e:
            tb = traceback.format_exc()
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 4px;")
            self.status_label.setText(f"Failed to save pipeline: {e}\n\n{tb}")

    def on_show(self):
        # Clear status if no thread is active
        is_running = False
        if hasattr(self, 'analysis_thread') and self.analysis_thread is not None:
            try:
                is_running = self.analysis_thread.isRunning()
            except RuntimeError:
                self.analysis_thread = None
                
        if not is_running:
            self.status_label.setText("")

    def on_hide(self):
        # Disconnect signals to prevent UI update if page is hidden
        if hasattr(self, 'analysis_thread') and self.analysis_thread is not None:
            try:
                if self.analysis_thread.isRunning():
                    try:
                        self.analysis_thread.status.disconnect()
                        self.analysis_thread.finished.disconnect()
                    except Exception:
                        pass
            except RuntimeError:
                self.analysis_thread = None
