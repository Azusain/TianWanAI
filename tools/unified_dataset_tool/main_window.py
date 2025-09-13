import sys
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, 
                            QFileDialog, QProgressBar, QComboBox, QSpinBox, 
                            QGroupBox, QFormLayout, QScrollArea, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon
from typing import Optional
import json

from dataset_manager import DatasetManager

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class AnalysisTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.dataset_manager = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # dataset path selection
        path_group = QGroupBox("dataset path")
        path_layout = QFormLayout()
        
        self.path_edit = QLineEdit()
        self.path_browse_btn = QPushButton("browse...")
        self.path_browse_btn.clicked.connect(self.browse_dataset_path)
        
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.path_browse_btn)
        path_layout.addRow("dataset directory:", path_row)
        
        # optional custom directories
        self.images_dir_edit = QLineEdit()
        self.images_browse_btn = QPushButton("browse...")
        self.images_browse_btn.clicked.connect(lambda: self.browse_directory(self.images_dir_edit))
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_dir_edit)
        images_row.addWidget(self.images_browse_btn)
        path_layout.addRow("images directory (optional):", images_row)
        
        self.labels_dir_edit = QLineEdit()
        self.labels_browse_btn = QPushButton("browse...")
        self.labels_browse_btn.clicked.connect(lambda: self.browse_directory(self.labels_dir_edit))
        labels_row = QHBoxLayout()
        labels_row.addWidget(self.labels_dir_edit)
        labels_row.addWidget(self.labels_browse_btn)
        path_layout.addRow("labels directory (optional):", labels_row)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # analyze button
        self.analyze_btn = QPushButton("analyze dataset")
        self.analyze_btn.clicked.connect(self.analyze_dataset)
        layout.addWidget(self.analyze_btn)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results display
        results_group = QGroupBox("analysis results")
        results_layout = QVBoxLayout()
        
        # summary stats
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["metric", "value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.stats_table)
        
        # class distribution
        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["class id", "class name", "count"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.class_table)
        
        # issues
        self.issues_text = QTextEdit()
        self.issues_text.setMaximumHeight(150)
        results_layout.addWidget(QLabel("dataset issues:"))
        results_layout.addWidget(self.issues_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.setLayout(layout)
    
    def browse_dataset_path(self):
        directory = QFileDialog.getExistingDirectory(self, "select dataset directory")
        if directory:
            self.path_edit.setText(directory)
    
    def browse_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "select directory")
        if directory:
            line_edit.setText(directory)
    
    def analyze_dataset(self):
        dataset_path = self.path_edit.text().strip()
        if not dataset_path:
            self.parent.show_error("please select a dataset directory")
            return
        
        try:
            self.dataset_manager = DatasetManager(dataset_path)
        except Exception as e:
            self.parent.show_error(f"invalid dataset path: {e}")
            return
        
        # get optional directories
        images_dir = self.images_dir_edit.text().strip() or None
        labels_dir = self.labels_dir_edit.text().strip() or None
        
        # start analysis in worker thread
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate
        
        self.worker = WorkerThread(self.dataset_manager.analyze_dataset, images_dir, labels_dir)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
    
    def on_analysis_complete(self, analysis):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.display_results(analysis)
    
    def on_analysis_error(self, error_msg):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.parent.show_error(f"analysis failed: {error_msg}")
    
    def display_results(self, analysis):
        # clear previous results
        self.stats_table.setRowCount(0)
        self.class_table.setRowCount(0)
        self.issues_text.clear()
        
        # display summary statistics
        stats = [
            ("total images", str(analysis.total_images)),
            ("labeled images", str(analysis.labeled_images)),
            ("unlabeled images", str(analysis.unlabeled_images)),
            ("total annotations", str(analysis.total_annotations)),
            ("unique classes", str(len(analysis.class_distribution))),
        ]
        
        self.stats_table.setRowCount(len(stats))
        for i, (metric, value) in enumerate(stats):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))
        
        # display class distribution
        if analysis.class_distribution:
            class_items = sorted(analysis.class_distribution.items())
            self.class_table.setRowCount(len(class_items))
            
            for i, (class_id, count) in enumerate(class_items):
                class_name = analysis.class_names.get(class_id, f"class_{class_id}")
                self.class_table.setItem(i, 0, QTableWidgetItem(str(class_id)))
                self.class_table.setItem(i, 1, QTableWidgetItem(class_name))
                self.class_table.setItem(i, 2, QTableWidgetItem(str(count)))
        
        # display issues
        issues_text = []
        if analysis.issues.missing_labels:
            issues_text.append(f"missing labels ({len(analysis.issues.missing_labels)} files):")
            for file_path in analysis.issues.missing_labels[:10]:  # show first 10
                issues_text.append(f"  - {file_path}")
            if len(analysis.issues.missing_labels) > 10:
                issues_text.append(f"  ... and {len(analysis.issues.missing_labels) - 10} more")
            issues_text.append("")
        
        if analysis.issues.empty_label_files:
            issues_text.append(f"empty label files ({len(analysis.issues.empty_label_files)} files):")
            for file_path in analysis.issues.empty_label_files[:10]:
                issues_text.append(f"  - {file_path}")
            if len(analysis.issues.empty_label_files) > 10:
                issues_text.append(f"  ... and {len(analysis.issues.empty_label_files) - 10} more")
            issues_text.append("")
        
        if analysis.issues.invalid_annotations:
            issues_text.append(f"invalid annotations ({len(analysis.issues.invalid_annotations)} files):")
            for file_path in analysis.issues.invalid_annotations[:10]:
                issues_text.append(f"  - {file_path}")
            if len(analysis.issues.invalid_annotations) > 10:
                issues_text.append(f"  ... and {len(analysis.issues.invalid_annotations) - 10} more")
        
        if not issues_text:
            issues_text = ["no issues found! 🎉"]
        
        self.issues_text.setText("\n".join(issues_text))

class SplitTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # input configuration
        input_group = QGroupBox("input configuration")
        input_layout = QFormLayout()
        
        self.dataset_path_edit = QLineEdit()
        self.dataset_browse_btn = QPushButton("browse...")
        self.dataset_browse_btn.clicked.connect(self.browse_dataset_path)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_path_edit)
        dataset_row.addWidget(self.dataset_browse_btn)
        input_layout.addRow("dataset directory:", dataset_row)
        
        # optional directories
        self.images_dir_edit = QLineEdit()
        self.images_browse_btn = QPushButton("browse...")
        self.images_browse_btn.clicked.connect(lambda: self.browse_directory(self.images_dir_edit))
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_dir_edit)
        images_row.addWidget(self.images_browse_btn)
        input_layout.addRow("images directory (optional):", images_row)
        
        self.labels_dir_edit = QLineEdit()
        self.labels_browse_btn = QPushButton("browse...")
        self.labels_browse_btn.clicked.connect(lambda: self.browse_directory(self.labels_dir_edit))
        labels_row = QHBoxLayout()
        labels_row.addWidget(self.labels_dir_edit)
        labels_row.addWidget(self.labels_browse_btn)
        input_layout.addRow("labels directory (optional):", labels_row)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # split configuration
        split_group = QGroupBox("split configuration")
        split_layout = QFormLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_browse_btn = QPushButton("browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_path)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_path_edit)
        output_row.addWidget(self.output_browse_btn)
        split_layout.addRow("output directory:", output_row)
        
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.1)
        self.train_ratio_spin.setSuffix(" (80% train, 20% val)")
        self.train_ratio_spin.valueChanged.connect(self.update_ratio_label)
        split_layout.addRow("train ratio:", self.train_ratio_spin)
        
        self.split_mode_combo = QComboBox()
        self.split_mode_combo.addItems(["random"])
        split_layout.addRow("split mode:", self.split_mode_combo)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(42)
        split_layout.addRow("random seed:", self.seed_spin)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # split button
        self.split_btn = QPushButton("split dataset")
        self.split_btn.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_btn)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        layout.addWidget(QLabel("split results:"))
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
    
    def browse_dataset_path(self):
        directory = QFileDialog.getExistingDirectory(self, "select dataset directory")
        if directory:
            self.dataset_path_edit.setText(directory)
    
    def browse_directory(self, line_edit):
        directory = QFileDialog.getExistingDirectory(self, "select directory")
        if directory:
            line_edit.setText(directory)
    
    def browse_output_path(self):
        directory = QFileDialog.getExistingDirectory(self, "select output directory")
        if directory:
            self.output_path_edit.setText(directory)
    
    def update_ratio_label(self):
        ratio = self.train_ratio_spin.value()
        train_pct = int(ratio * 100)
        val_pct = 100 - train_pct
        self.train_ratio_spin.setSuffix(f" ({train_pct}% train, {val_pct}% val)")
    
    def split_dataset(self):
        dataset_path = self.dataset_path_edit.text().strip()
        output_path = self.output_path_edit.text().strip()
        
        if not dataset_path or not output_path:
            self.parent.show_error("please specify both dataset and output directories")
            return
        
        try:
            dataset_manager = DatasetManager(dataset_path)
        except Exception as e:
            self.parent.show_error(f"invalid dataset path: {e}")
            return
        
        # get parameters
        train_ratio = self.train_ratio_spin.value()
        split_mode = self.split_mode_combo.currentText()
        seed = self.seed_spin.value()
        images_dir = self.images_dir_edit.text().strip() or None
        labels_dir = self.labels_dir_edit.text().strip() or None
        
        # start split in worker thread
        self.split_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.worker = WorkerThread(
            dataset_manager.split_dataset,
            output_path, train_ratio, split_mode, seed, images_dir, labels_dir
        )
        self.worker.finished.connect(self.on_split_complete)
        self.worker.error.connect(self.on_split_error)
        self.worker.start()
    
    def on_split_complete(self, summary):
        self.split_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.results_text.setText(summary)
    
    def on_split_error(self, error_msg):
        self.split_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.parent.show_error(f"split failed: {error_msg}")

class VisualizationTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # configuration
        config_group = QGroupBox("visualization configuration")
        config_layout = QFormLayout()
        
        self.dataset_path_edit = QLineEdit()
        self.dataset_browse_btn = QPushButton("browse...")
        self.dataset_browse_btn.clicked.connect(self.browse_dataset_path)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_path_edit)
        dataset_row.addWidget(self.dataset_browse_btn)
        config_layout.addRow("dataset directory:", dataset_row)
        
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(1, 50)
        self.sample_count_spin.setValue(5)
        config_layout.addRow("sample count:", self.sample_count_spin)
        
        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_path)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(self.output_browse_btn)
        config_layout.addRow("output directory (optional):", output_row)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # generate button
        self.generate_btn = QPushButton("generate visualizations")
        self.generate_btn.clicked.connect(self.generate_visualizations)
        layout.addWidget(self.generate_btn)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results
        self.results_text = QTextEdit()
        layout.addWidget(QLabel("generated files:"))
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
    
    def browse_dataset_path(self):
        directory = QFileDialog.getExistingDirectory(self, "select dataset directory")
        if directory:
            self.dataset_path_edit.setText(directory)
    
    def browse_output_path(self):
        directory = QFileDialog.getExistingDirectory(self, "select output directory")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def generate_visualizations(self):
        dataset_path = self.dataset_path_edit.text().strip()
        if not dataset_path:
            self.parent.show_error("please select a dataset directory")
            return
        
        try:
            dataset_manager = DatasetManager(dataset_path)
        except Exception as e:
            self.parent.show_error(f"invalid dataset path: {e}")
            return
        
        # get parameters
        sample_count = self.sample_count_spin.value()
        output_dir = self.output_dir_edit.text().strip() or None
        
        # start visualization in worker thread
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.worker = WorkerThread(
            dataset_manager.visualize_samples,
            output_dir, sample_count
        )
        self.worker.finished.connect(self.on_visualization_complete)
        self.worker.error.connect(self.on_visualization_error)
        self.worker.start()
    
    def on_visualization_complete(self, file_paths):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if file_paths:
            results_text = f"generated {len(file_paths)} visualization files:\n\n"
            for path in file_paths:
                results_text += f"- {path}\n"
        else:
            results_text = "no labeled images found for visualization"
        
        self.results_text.setText(results_text)
    
    def on_visualization_error(self, error_msg):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.parent.show_error(f"visualization failed: {error_msg}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("unified dataset tool")
        self.setMinimumSize(900, 700)
        
        # create central widget with tab layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # create tab widget
        tab_widget = QTabWidget()
        
        # add tabs
        self.analysis_tab = AnalysisTab(self)
        self.split_tab = SplitTab(self)
        self.visualization_tab = VisualizationTab(self)
        
        tab_widget.addTab(self.analysis_tab, "dataset analysis")
        tab_widget.addTab(self.split_tab, "dataset split")
        tab_widget.addTab(self.visualization_tab, "visualization")
        
        layout.addWidget(tab_widget)
        
        # status bar
        self.statusBar().showMessage("ready")
    
    def show_error(self, message):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "error", message)
