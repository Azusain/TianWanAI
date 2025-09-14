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
from video_processor import VideoProcessor

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
        layout.setSpacing(15)
        
        # dataset path selection
        path_group = QGroupBox("Dataset Path")
        path_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        path_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        path_layout = QFormLayout()
        path_layout.setVerticalSpacing(12)
        
        self.path_edit = QLineEdit()
        self.path_edit.setFont(QFont("Arial", 11))
        self.path_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        
        self.path_browse_btn = QPushButton("Browse...")
        self.path_browse_btn.setFont(QFont("Arial", 11))
        self.path_browse_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QPushButton:hover {
                background-color: #e9e9e9;
            }
            QPushButton:pressed {
                background-color: #d4edda;
            }
        """)
        self.path_browse_btn.clicked.connect(self.browse_dataset_path)
        
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.path_browse_btn)
        path_layout.addRow("dataset directory:", path_row)
        
        # optional custom directories
        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setFont(QFont("Arial", 11))
        self.images_dir_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        
        self.images_browse_btn = QPushButton("Browse...")
        self.images_browse_btn.setFont(QFont("Arial", 11))
        self.images_browse_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QPushButton:hover {
                background-color: #e9e9e9;
            }
            QPushButton:pressed {
                background-color: #d4edda;
            }
        """)
        self.images_browse_btn.clicked.connect(lambda: self.browse_directory(self.images_dir_edit))
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_dir_edit)
        images_row.addWidget(self.images_browse_btn)
        path_layout.addRow("images directory (optional):", images_row)
        
        self.labels_dir_edit = QLineEdit()
        self.labels_dir_edit.setFont(QFont("Arial", 11))
        self.labels_dir_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        
        self.labels_browse_btn = QPushButton("Browse...")
        self.labels_browse_btn.setFont(QFont("Arial", 11))
        self.labels_browse_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QPushButton:hover {
                background-color: #e9e9e9;
            }
            QPushButton:pressed {
                background-color: #d4edda;
            }
        """)
        self.labels_browse_btn.clicked.connect(lambda: self.browse_directory(self.labels_dir_edit))
        labels_row = QHBoxLayout()
        labels_row.addWidget(self.labels_dir_edit)
        labels_row.addWidget(self.labels_browse_btn)
        path_layout.addRow("labels directory (optional):", labels_row)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # analyze button
        self.analyze_btn = QPushButton("Analyze Dataset")
        self.analyze_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.analyze_btn.clicked.connect(self.analyze_dataset)
        layout.addWidget(self.analyze_btn)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results display
        results_group = QGroupBox("Analysis Results")
        results_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        results_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        results_layout = QVBoxLayout()
        results_layout.setSpacing(12)
        
        # summary stats
        stats_label = QLabel("Dataset Summary:")
        stats_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        results_layout.addWidget(stats_label)
        
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setFont(QFont("Arial", 11))
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.stats_table)
        
        # class distribution
        class_label = QLabel("Class Distribution:")
        class_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        results_layout.addWidget(class_label)
        
        self.class_table = QTableWidget(0, 3)
        self.class_table.setFont(QFont("Arial", 11))
        self.class_table.setHorizontalHeaderLabels(["Class ID", "Class Name", "Count"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.class_table)
        
        # issues
        issues_label = QLabel("Dataset Issues:")
        issues_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        results_layout.addWidget(issues_label)
        
        self.issues_text = QTextEdit()
        self.issues_text.setFont(QFont("Arial", 11))
        self.issues_text.setMaximumHeight(150)
        self.issues_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
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
            issues_text = ["no issues found!"]
        
        self.issues_text.setText("\n".join(issues_text))

class SplitTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # input configuration
        input_group = QGroupBox("Input Configuration")
        input_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        input_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        input_layout = QFormLayout()
        input_layout.setVerticalSpacing(12)
        
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
        split_group = QGroupBox("Split Configuration")
        split_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        split_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        split_layout = QFormLayout()
        split_layout.setVerticalSpacing(12)
        
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
        self.split_btn = QPushButton("Split Dataset")
        self.split_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.split_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.split_btn.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_btn)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results
        results_label = QLabel("Split Results:")
        results_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setFont(QFont("Arial", 11))
        self.results_text.setMaximumHeight(150)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
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
        layout.setSpacing(15)
        
        # configuration
        config_group = QGroupBox("Visualization Configuration")
        config_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        config_layout = QFormLayout()
        config_layout.setVerticalSpacing(12)
        
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
        self.generate_btn = QPushButton("Generate Visualizations")
        self.generate_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_visualizations)
        layout.addWidget(self.generate_btn)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results
        results_label = QLabel("Generated Files:")
        results_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setFont(QFont("Arial", 11))
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
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

class VideoProcessingTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.video_processor = VideoProcessor()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # input configuration
        input_group = QGroupBox("Input Configuration")
        input_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        input_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        input_layout = QFormLayout()
        input_layout.setVerticalSpacing(12)
        
        # video file selection
        self.video_path_edit = QLineEdit()
        self.video_browse_btn = QPushButton("browse...")
        self.video_browse_btn.clicked.connect(self.browse_video_file)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_path_edit)
        video_row.addWidget(self.video_browse_btn)
        input_layout.addRow("video file:", video_row)
        
        # batch directory
        self.batch_dir_edit = QLineEdit()
        self.batch_browse_btn = QPushButton("browse...")
        self.batch_browse_btn.clicked.connect(self.browse_batch_directory)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_dir_edit)
        batch_row.addWidget(self.batch_browse_btn)
        input_layout.addRow("batch directory (optional):", batch_row)
        
        # output directory
        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_directory)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(self.output_browse_btn)
        input_layout.addRow("output directory:", output_row)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # extraction settings
        settings_group = QGroupBox("Extraction Settings")
        settings_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        settings_layout = QFormLayout()
        settings_layout.setVerticalSpacing(12)
        
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 1000)
        self.frame_interval_spin.setValue(30)
        self.frame_interval_spin.setSuffix(" frames")
        settings_layout.addRow("frame interval:", self.frame_interval_spin)
        
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 10000)
        self.max_frames_spin.setValue(0)
        self.max_frames_spin.setSuffix(" (0 = unlimited)")
        settings_layout.addRow("max frames:", self.max_frames_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # action buttons
        buttons_layout = QHBoxLayout()
        
        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.extract_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.extract_btn.clicked.connect(self.extract_frames)
        buttons_layout.addWidget(self.extract_btn)
        
        self.info_btn = QPushButton("Get Video Info")
        self.info_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.info_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.info_btn.clicked.connect(self.get_video_info)
        buttons_layout.addWidget(self.info_btn)
        
        layout.addLayout(buttons_layout)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # results
        results_label = QLabel("Processing Results:")
        results_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setFont(QFont("Arial", 11))
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
    
    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "select video file", "",
            "video files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;all files (*)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)
    
    def browse_batch_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "select batch directory")
        if directory:
            self.batch_dir_edit.setText(directory)
    
    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "select output directory")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def extract_frames(self):
        video_path = self.video_path_edit.text().strip()
        batch_dir = self.batch_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        
        if not output_dir:
            self.parent.show_error("please specify an output directory")
            return
        
        if not video_path and not batch_dir:
            self.parent.show_error("please specify either a video file or batch directory")
            return
        
        # get settings
        frame_interval = self.frame_interval_spin.value()
        max_frames = self.max_frames_spin.value() if self.max_frames_spin.value() > 0 else None
        
        # start extraction in worker thread
        self.extract_btn.setEnabled(False)
        self.info_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        if batch_dir:
            func = self.video_processor.process_batch
            args = (batch_dir, output_dir, frame_interval, max_frames)
        else:
            func = self.video_processor.extract_frames
            args = (video_path, output_dir, frame_interval, max_frames)
        
        self.worker = WorkerThread(func, *args)
        self.worker.finished.connect(self.on_extraction_complete)
        self.worker.error.connect(self.on_extraction_error)
        self.worker.start()
    
    def get_video_info(self):
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            self.parent.show_error("please select a video file")
            return
        
        try:
            info = self.video_processor.get_video_info(video_path)
            info_text = f"video information for: {video_path}\n\n"
            for key, value in info.items():
                info_text += f"{key}: {value}\n"
            self.results_text.setText(info_text)
        except Exception as e:
            self.parent.show_error(f"failed to get video info: {e}")
    
    def on_extraction_complete(self, result):
        self.extract_btn.setEnabled(True)
        self.info_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if isinstance(result, dict) and 'summary' in result:
            self.results_text.setText(result['summary'])
        else:
            self.results_text.setText(f"frame extraction completed: {result}")
    
    def on_extraction_error(self, error_msg):
        self.extract_btn.setEnabled(True)
        self.info_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.parent.show_error(f"frame extraction failed: {error_msg}")

# Data Processing Tab - combines analysis, split, and visualization
class DataProcessingTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        # Create sub-tabs for data processing functions
        sub_tabs = QTabWidget()
        
        # Add existing tabs as sub-tabs
        analysis_tab = AnalysisTab(self.parent)
        split_tab = SplitTab(self.parent) 
        visualization_tab = VisualizationTab(self.parent)
        
        sub_tabs.addTab(analysis_tab, "Analysis")
        sub_tabs.addTab(split_tab, "Split")
        sub_tabs.addTab(visualization_tab, "Visualization")
        
        layout.addWidget(sub_tabs)
        self.setLayout(layout)


# Format Conversion Tab
class FormatConversionTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # VOC to YOLO conversion
        voc_group = QGroupBox("PASCAL VOC to YOLO Conversion")
        voc_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        voc_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        voc_layout = QFormLayout()
        voc_layout.setVerticalSpacing(12)
        
        # XML labels directory
        self.xml_dir_edit = QLineEdit()
        self.xml_dir_edit.setFont(QFont("Arial", 11))
        self.xml_dir_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        
        self.xml_browse_btn = QPushButton("Browse...")
        self.xml_browse_btn.setFont(QFont("Arial", 11))
        self.xml_browse_btn.clicked.connect(self.browse_xml_directory)
        xml_row = QHBoxLayout()
        xml_row.addWidget(self.xml_dir_edit)
        xml_row.addWidget(self.xml_browse_btn)
        voc_layout.addRow("XML Labels Directory:", xml_row)
        
        # Classes file
        self.classes_file_edit = QLineEdit()
        self.classes_file_edit.setFont(QFont("Arial", 11))
        self.classes_file_edit.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
        
        self.classes_browse_btn = QPushButton("Browse...")
        self.classes_browse_btn.setFont(QFont("Arial", 11))
        self.classes_browse_btn.clicked.connect(self.browse_classes_file)
        classes_row = QHBoxLayout()
        classes_row.addWidget(self.classes_file_edit)
        classes_row.addWidget(self.classes_browse_btn)
        voc_layout.addRow("Classes File (classes.names):", classes_row)
        
        voc_group.setLayout(voc_layout)
        layout.addWidget(voc_group)
        
        # Convert button
        self.convert_btn = QPushButton("Convert VOC to YOLO")
        self.convert_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.convert_btn.clicked.connect(self.convert_voc_to_yolo)
        layout.addWidget(self.convert_btn)
        
        # Results
        results_label = QLabel("Conversion Results:")
        results_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(results_label)
        
        self.conversion_results = QTextEdit()
        self.conversion_results.setFont(QFont("Arial", 11))
        self.conversion_results.setMaximumHeight(150)
        self.conversion_results.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.conversion_results)
        
        self.setLayout(layout)
    
    def browse_xml_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select XML Labels Directory")
        if directory:
            self.xml_dir_edit.setText(directory)
    
    def browse_classes_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Classes File", "",
            "Text files (*.names *.txt);;All files (*)"
        )
        if file_path:
            self.classes_file_edit.setText(file_path)
    
    def convert_voc_to_yolo(self):
        xml_dir = self.xml_dir_edit.text().strip()
        classes_file = self.classes_file_edit.text().strip()
        
        if not xml_dir or not classes_file:
            self.parent.show_error("Please specify both XML directory and classes file")
            return
        
        try:
            self.convert_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_voc_conversion, xml_dir, classes_file)
            self.worker.finished.connect(self.on_conversion_complete)
            self.worker.error.connect(self.on_conversion_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Conversion failed: {e}")
    
    def _do_voc_conversion(self, xml_dir, classes_file):
        import glob
        import xml.etree.ElementTree as ET
        import os
        
        # Load classes
        classes_dict = {}
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                class_name = line.strip()
                classes_dict[class_name] = idx
        
        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
        converted_count = 0
        
        for xml_file in xml_files:
            try:
                # Parse XML
                tree = ET.parse(xml_file)
                size = tree.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                lines = []
                for obj in tree.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in classes_dict:
                        continue
                        
                    label = classes_dict[class_name]
                    bbox = obj.find('bndbox')
                    x, y, x2, y2 = (
                        int(bbox.find('xmin').text),
                        int(bbox.find('ymin').text),
                        int(bbox.find('xmax').text),
                        int(bbox.find('ymax').text)
                    )
                    
                    # Convert to YOLO format
                    cx = (x2 + x) * 0.5 / width
                    cy = (y2 + y) * 0.5 / height
                    w = (x2 - x) * 1. / width
                    h = (y2 - y) * 1. / height
                    
                    line = f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    lines.append(line)
                
                # Save YOLO format
                txt_file = xml_file.replace(".xml", ".txt")
                with open(txt_file, "w") as f:
                    f.writelines(lines)
                
                converted_count += 1
                
            except Exception as e:
                print(f"Error converting {xml_file}: {e}")
        
        return f"Successfully converted {converted_count} XML files to YOLO format"
    
    def on_conversion_complete(self, result):
        self.convert_btn.setEnabled(True)
        self.conversion_results.setText(result)
    
    def on_conversion_error(self, error_msg):
        self.convert_btn.setEnabled(True)
        self.parent.show_error(f"Conversion failed: {error_msg}")

# Dataset Management Tab - dataset reduction, completion, and class replacement
class DatasetManagementTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Dataset Reduction Tool
        reduce_group = QGroupBox("Dataset Reduction Tool")
        reduce_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        reduce_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        reduce_layout = QFormLayout()
        reduce_layout.setVerticalSpacing(12)
        
        # Dataset directory
        self.reduce_dataset_edit = QLineEdit()
        self.reduce_dataset_edit.setFont(QFont("Arial", 11))
        self.reduce_dataset_browse = QPushButton("Browse...")
        self.reduce_dataset_browse.setFont(QFont("Arial", 11))
        self.reduce_dataset_browse.clicked.connect(self.browse_reduce_dataset)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.reduce_dataset_edit)
        dataset_row.addWidget(self.reduce_dataset_browse)
        reduce_layout.addRow("Dataset Directory:", dataset_row)
        
        # Class to reduce
        self.class_id_spin = QSpinBox()
        self.class_id_spin.setRange(0, 100)
        self.class_id_spin.setValue(1)
        reduce_layout.addRow("Class ID to Reduce:", self.class_id_spin)
        
        # Reduction percentage
        self.reduction_spin = QDoubleSpinBox()
        self.reduction_spin.setRange(0.1, 0.9)
        self.reduction_spin.setValue(0.6)
        self.reduction_spin.setSuffix(" (60% to remove)")
        reduce_layout.addRow("Reduction Ratio:", self.reduction_spin)
        
        reduce_group.setLayout(reduce_layout)
        layout.addWidget(reduce_group)
        
        # Dataset Completion Tool
        complete_group = QGroupBox("Dataset Completion Tool")
        complete_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        complete_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        complete_layout = QFormLayout()
        complete_layout.setVerticalSpacing(12)
        
        # Dataset directory for completion
        self.complete_dataset_edit = QLineEdit()
        self.complete_dataset_edit.setFont(QFont("Arial", 11))
        self.complete_dataset_browse = QPushButton("Browse...")
        self.complete_dataset_browse.setFont(QFont("Arial", 11))
        self.complete_dataset_browse.clicked.connect(self.browse_complete_dataset)
        complete_dataset_row = QHBoxLayout()
        complete_dataset_row.addWidget(self.complete_dataset_edit)
        complete_dataset_row.addWidget(self.complete_dataset_browse)
        complete_layout.addRow("Dataset Directory:", complete_dataset_row)
        
        complete_group.setLayout(complete_layout)
        layout.addWidget(complete_group)
        
        # Class Replacement Tool
        class_replace_group = QGroupBox("Class ID Replacement Tool")
        class_replace_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        class_replace_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)
        class_replace_layout = QFormLayout()
        class_replace_layout.setVerticalSpacing(12)
        
        # Dataset directory for class replacement
        self.replace_dataset_edit = QLineEdit()
        self.replace_dataset_edit.setFont(QFont("Arial", 11))
        self.replace_dataset_browse = QPushButton("Browse...")
        self.replace_dataset_browse.setFont(QFont("Arial", 11))
        self.replace_dataset_browse.clicked.connect(self.browse_replace_dataset)
        replace_dataset_row = QHBoxLayout()
        replace_dataset_row.addWidget(self.replace_dataset_edit)
        replace_dataset_row.addWidget(self.replace_dataset_browse)
        class_replace_layout.addRow("Dataset Directory:", replace_dataset_row)
        
        # New class ID
        self.new_class_id_spin = QSpinBox()
        self.new_class_id_spin.setRange(0, 100)
        self.new_class_id_spin.setValue(0)
        class_replace_layout.addRow("New Class ID:", self.new_class_id_spin)
        
        class_replace_group.setLayout(class_replace_layout)
        layout.addWidget(class_replace_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        # Reduce button
        self.reduce_btn = QPushButton("Reduce Dataset")
        self.reduce_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.reduce_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.reduce_btn.clicked.connect(self.reduce_dataset)
        buttons_layout.addWidget(self.reduce_btn)
        
        # Complete dataset button
        self.complete_btn = QPushButton("Complete Dataset")
        self.complete_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.complete_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.complete_btn.clicked.connect(self.complete_dataset)
        buttons_layout.addWidget(self.complete_btn)
        
        # Replace class IDs button
        self.replace_btn = QPushButton("Replace Class IDs")
        self.replace_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.replace_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.replace_btn.clicked.connect(self.replace_class_ids)
        buttons_layout.addWidget(self.replace_btn)
        
        layout.addLayout(buttons_layout)
        
        # Results
        results_label = QLabel("Operation Results:")
        results_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(results_label)
        
        self.advanced_results = QTextEdit()
        self.advanced_results.setFont(QFont("Arial", 11))
        self.advanced_results.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.advanced_results)
        
        self.setLayout(layout)
    
    def browse_reduce_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self.reduce_dataset_edit.setText(directory)
    
    def reduce_dataset(self):
        dataset_dir = self.reduce_dataset_edit.text().strip()
        class_id = self.class_id_spin.value()
        reduction_ratio = self.reduction_spin.value()
        
        if not dataset_dir:
            self.parent.show_error("Please specify dataset directory")
            return
        
        try:
            self.reduce_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_dataset_reduction, dataset_dir, class_id, reduction_ratio)
            self.worker.finished.connect(self.on_reduction_complete)
            self.worker.error.connect(self.on_reduction_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Reduction failed: {e}")
    
    def _do_dataset_reduction(self, dataset_dir, class_id, reduction_ratio):
        import os
        import random
        
        removed_count = 0
        total_count = 0
        
        # Process labels directory
        labels_dir = os.path.join(dataset_dir, 'labels')
        images_dir = os.path.join(dataset_dir, 'images')
        
        if not os.path.exists(labels_dir):
            raise Exception(f"Labels directory not found: {labels_dir}")
        
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, filename)
            
            try:
                with open(label_path, 'r') as f:
                    first_char = f.read(1)
                    if first_char and int(first_char) == class_id:
                        total_count += 1
                        if random.random() <= reduction_ratio:
                            # Remove both label and image files
                            os.remove(label_path)
                            
                            # Try different image extensions
                            base_name = filename.replace('.txt', '')
                            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                                img_path = os.path.join(images_dir, base_name + ext)
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                                    break
                            
                            removed_count += 1
            except Exception as e:
                print(f"Error processing {label_path}: {e}")
        
        return f"Removed {removed_count} out of {total_count} files with class ID {class_id} ({reduction_ratio*100:.1f}% reduction ratio)"
    
    def on_reduction_complete(self, result):
        self.reduce_btn.setEnabled(True)
        self.advanced_results.setText(result)
    
    def on_reduction_error(self, error_msg):
        self.reduce_btn.setEnabled(True)
        self.parent.show_error(f"Reduction failed: {error_msg}")
    
    def browse_complete_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory for Completion")
        if directory:
            self.complete_dataset_edit.setText(directory)
    
    def browse_replace_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory for Class Replacement")
        if directory:
            self.replace_dataset_edit.setText(directory)
    
    def complete_dataset(self):
        """Create empty label files for images that don't have corresponding labels"""
        dataset_dir = self.complete_dataset_edit.text().strip()
        
        if not dataset_dir:
            self.parent.show_error("Please specify dataset directory")
            return
        
        try:
            self.complete_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_dataset_completion, dataset_dir)
            self.worker.finished.connect(self.on_completion_complete)
            self.worker.error.connect(self.on_completion_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Dataset completion failed: {e}")
    
    def _do_dataset_completion(self, dataset_dir):
        """Create empty txt files for images without corresponding labels"""
        import os
        
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        
        if not os.path.exists(images_dir):
            raise Exception(f"Images directory not found: {images_dir}")
        
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        
        created_count = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Get all image files
        for filename in os.listdir(images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Get the base name without extension
                base_name = os.path.splitext(filename)[0]
                label_path = os.path.join(labels_dir, base_name + '.txt')
                
                # Create empty label file if it doesn't exist
                if not os.path.exists(label_path):
                    with open(label_path, 'w') as f:
                        pass  # Create empty file
                    created_count += 1
        
        return f"Created {created_count} empty label files for unlabeled images"
    
    def on_completion_complete(self, result):
        self.complete_btn.setEnabled(True)
        self.advanced_results.setText(result)
    
    def on_completion_error(self, error_msg):
        self.complete_btn.setEnabled(True)
        self.parent.show_error(f"Dataset completion failed: {error_msg}")
    
    def replace_class_ids(self):
        """Replace all class IDs in label files with a specified new class ID"""
        dataset_dir = self.replace_dataset_edit.text().strip()
        new_class_id = self.new_class_id_spin.value()
        
        if not dataset_dir:
            self.parent.show_error("Please specify dataset directory")
            return
        
        try:
            self.replace_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_class_replacement, dataset_dir, new_class_id)
            self.worker.finished.connect(self.on_replacement_complete)
            self.worker.error.connect(self.on_replacement_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Class replacement failed: {e}")
    
    def _do_class_replacement(self, dataset_dir, new_class_id):
        """Replace all class IDs in YOLO label files with the specified new class ID"""
        import os
        import re
        
        labels_dir = os.path.join(dataset_dir, 'labels')
        
        if not os.path.exists(labels_dir):
            raise Exception(f"Labels directory not found: {labels_dir}")
        
        processed_files = 0
        total_annotations = 0
        
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, filename)
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:  # Skip empty files
                    continue
                    
                modified_lines = []
                file_modified = False
                
                for line in lines:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        modified_lines.append(line)
                        continue
                    
                    # Parse YOLO format: class_id x_center y_center width height
                    parts = line.split()
                    if len(parts) >= 5:
                        # Replace class ID with new class ID
                        old_class_id = parts[0]
                        parts[0] = str(new_class_id)
                        modified_line = ' '.join(parts)
                        modified_lines.append(modified_line)
                        
                        if old_class_id != str(new_class_id):
                            file_modified = True
                        total_annotations += 1
                    else:
                        # Keep invalid lines as is (shouldn't happen in proper YOLO format)
                        modified_lines.append(line)
                
                # Write back to file if modified
                if file_modified or len(modified_lines) != len(lines):
                    with open(label_path, 'w') as f:
                        for line in modified_lines:
                            if line.strip():  # Only write non-empty lines
                                f.write(line + '\n')
                    processed_files += 1
                    
            except Exception as e:
                print(f"Error processing {label_path}: {e}")
        
        return f"Processed {processed_files} label files, updated {total_annotations} annotations to class ID {new_class_id}"
    
    def on_replacement_complete(self, result):
        self.replace_btn.setEnabled(True)
        self.advanced_results.setText(result)
    
    def on_replacement_error(self, error_msg):
        self.replace_btn.setEnabled(True)
        self.parent.show_error(f"Class replacement failed: {error_msg}")

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
        self.video_processing_tab = VideoProcessingTab(self)
        
        # reorganize tabs into logical groups
        self.data_processing_tab = DataProcessingTab(self)
        self.dataset_management_tab = DatasetManagementTab(self)
        self.format_conversion_tab = FormatConversionTab(self)
        self.video_processing_tab = VideoProcessingTab(self)
        
        tab_widget.addTab(self.data_processing_tab, "Data Processing")
        tab_widget.addTab(self.dataset_management_tab, "Dataset Management")
        tab_widget.addTab(self.format_conversion_tab, "Format Conversion")
        tab_widget.addTab(self.video_processing_tab, "Video Processing")
        
        layout.addWidget(tab_widget)
        
        # status bar
        self.statusBar().showMessage("ready")
    
    def show_error(self, message):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "error", message)

# Main entry point
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
