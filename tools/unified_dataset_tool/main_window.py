import sys
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, 
                            QFileDialog, QProgressBar, QComboBox, QSpinBox, 
                            QGroupBox, QFormLayout, QScrollArea, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QCheckBox, QDoubleSpinBox,
                            QListWidget)
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
        
        # train ratio
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setValue(0.7)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setDecimals(2)
        self.train_ratio_spin.valueChanged.connect(self.update_ratio_labels)
        split_layout.addRow("train ratio:", self.train_ratio_spin)
        
        # val ratio
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.05, 0.5)
        self.val_ratio_spin.setValue(0.15)
        self.val_ratio_spin.setSingleStep(0.05)
        self.val_ratio_spin.setDecimals(2)
        self.val_ratio_spin.valueChanged.connect(self.update_ratio_labels)
        split_layout.addRow("val ratio:", self.val_ratio_spin)
        
        # test ratio
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.05, 0.5)
        self.test_ratio_spin.setValue(0.15)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setDecimals(2)
        self.test_ratio_spin.valueChanged.connect(self.update_ratio_labels)
        split_layout.addRow("test ratio:", self.test_ratio_spin)
        
        # ratio summary label
        self.ratio_summary_label = QLabel()
        self.ratio_summary_label.setFont(QFont("Arial", 10))
        self.ratio_summary_label.setStyleSheet("color: #666; padding: 5px;")
        split_layout.addRow("", self.ratio_summary_label)
        self.update_ratio_labels()  # initialize label
        
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
    
    def update_ratio_labels(self):
        """update ratio summary label when any ratio changes"""
        train_ratio = self.train_ratio_spin.value()
        val_ratio = self.val_ratio_spin.value()
        test_ratio = self.test_ratio_spin.value()
        
        total = train_ratio + val_ratio + test_ratio
        
        # update summary label
        train_pct = train_ratio * 100
        val_pct = val_ratio * 100
        test_pct = test_ratio * 100
        
        if abs(total - 1.0) > 0.001:
            self.ratio_summary_label.setText(f"⚠ ratios sum to {total:.3f} (should be 1.0)")
            self.ratio_summary_label.setStyleSheet("color: red; padding: 5px; font-weight: bold;")
        else:
            summary = f"train: {train_pct:.1f}%, val: {val_pct:.1f}%, test: {test_pct:.1f}%"
            self.ratio_summary_label.setText(f"✓ {summary}")
            self.ratio_summary_label.setStyleSheet("color: #666; padding: 5px;")
    
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
        val_ratio = self.val_ratio_spin.value()
        test_ratio = self.test_ratio_spin.value()
        split_mode = self.split_mode_combo.currentText()
        seed = self.seed_spin.value()
        images_dir = self.images_dir_edit.text().strip() or None
        labels_dir = self.labels_dir_edit.text().strip() or None
        
        # validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            self.parent.show_error(f"ratios must sum to 1.0, currently {total:.3f}")
            return
        
        # start split in worker thread
        self.split_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.worker = WorkerThread(
            dataset_manager.split_dataset,
            output_path, train_ratio, val_ratio, test_ratio, split_mode, seed, images_dir, labels_dir
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
        
        # Create sub-tabs for data-related functions
        sub_tabs = QTabWidget()
        
        # Add existing tabs as sub-tabs
        analysis_tab = AnalysisTab(self.parent)
        split_tab = SplitTab(self.parent) 
        visualization_tab = VisualizationTab(self.parent)
        dataset_mgmt_tab = DatasetManagementTab(self.parent)
        format_conv_tab = FormatConversionTab(self.parent)
        
        sub_tabs.addTab(analysis_tab, "Analysis")
        sub_tabs.addTab(split_tab, "Split")
        sub_tabs.addTab(visualization_tab, "Visualization")
        sub_tabs.addTab(dataset_mgmt_tab, "Management")
        sub_tabs.addTab(format_conv_tab, "Conversion")
        
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
        voc_layout.addRow("Classes File (optional - auto-detect if empty):", classes_row)
        
        voc_group.setLayout(voc_layout)
        layout.addWidget(voc_group)
        
        # YOLO to VOC conversion
        yolo_group = QGroupBox("YOLO to PASCAL VOC Conversion")
        yolo_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        yolo_group.setStyleSheet("""
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
        yolo_layout = QFormLayout()
        yolo_layout.setVerticalSpacing(12)
        
        # YOLO dataset directory
        self.yolo_dataset_edit = QLineEdit()
        self.yolo_dataset_edit.setFont(QFont("Arial", 11))
        self.yolo_dataset_edit.setStyleSheet("""
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
        
        self.yolo_dataset_browse = QPushButton("Browse...")
        self.yolo_dataset_browse.setFont(QFont("Arial", 11))
        self.yolo_dataset_browse.clicked.connect(self.browse_yolo_dataset)
        yolo_dataset_row = QHBoxLayout()
        yolo_dataset_row.addWidget(self.yolo_dataset_edit)
        yolo_dataset_row.addWidget(self.yolo_dataset_browse)
        yolo_layout.addRow("YOLO Dataset Directory:", yolo_dataset_row)
        
        # Classes file for YOLO to VOC
        self.yolo_classes_file_edit = QLineEdit()
        self.yolo_classes_file_edit.setFont(QFont("Arial", 11))
        self.yolo_classes_file_edit.setStyleSheet("""
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
        
        self.yolo_classes_browse = QPushButton("Browse...")
        self.yolo_classes_browse.setFont(QFont("Arial", 11))
        self.yolo_classes_browse.clicked.connect(self.browse_yolo_classes_file)
        yolo_classes_row = QHBoxLayout()
        yolo_classes_row.addWidget(self.yolo_classes_file_edit)
        yolo_classes_row.addWidget(self.yolo_classes_browse)
        yolo_layout.addRow("Classes File (classes.txt):", yolo_classes_row)
        
        yolo_group.setLayout(yolo_layout)
        layout.addWidget(yolo_group)
        
        # YOLO to Flat Format conversion
        flat_group = QGroupBox("YOLO to Flat Format Conversion")
        flat_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        flat_group.setStyleSheet("""
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
        flat_layout = QFormLayout()
        flat_layout.setVerticalSpacing(12)
        
        # YOLO input directory
        self.flat_input_edit = QLineEdit()
        self.flat_input_edit.setFont(QFont("Arial", 11))
        self.flat_input_edit.setStyleSheet("""
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
        
        self.flat_input_browse = QPushButton("Browse...")
        self.flat_input_browse.setFont(QFont("Arial", 11))
        self.flat_input_browse.clicked.connect(self.browse_flat_input)
        flat_input_row = QHBoxLayout()
        flat_input_row.addWidget(self.flat_input_edit)
        flat_input_row.addWidget(self.flat_input_browse)
        flat_layout.addRow("YOLO Dataset Directory:", flat_input_row)
        
        # Flat output directory
        self.flat_output_edit = QLineEdit()
        self.flat_output_edit.setFont(QFont("Arial", 11))
        self.flat_output_edit.setStyleSheet("""
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
        
        self.flat_output_browse = QPushButton("Browse...")
        self.flat_output_browse.setFont(QFont("Arial", 11))
        self.flat_output_browse.clicked.connect(self.browse_flat_output)
        flat_output_row = QHBoxLayout()
        flat_output_row.addWidget(self.flat_output_edit)
        flat_output_row.addWidget(self.flat_output_browse)
        flat_layout.addRow("Output Directory:", flat_output_row)
        
        # Conversion options
        self.merge_splits_checkbox = QCheckBox("Merge splits into single folder")
        self.merge_splits_checkbox.setFont(QFont("Arial", 11))
        self.merge_splits_checkbox.setChecked(True)
        flat_layout.addRow("", self.merge_splits_checkbox)
        
        self.preserve_split_info_checkbox = QCheckBox("Preserve split info in filenames")
        self.preserve_split_info_checkbox.setFont(QFont("Arial", 11))
        self.preserve_split_info_checkbox.setChecked(True)
        flat_layout.addRow("", self.preserve_split_info_checkbox)
        
        flat_group.setLayout(flat_layout)
        layout.addWidget(flat_group)
        
        # Convert buttons
        buttons_layout = QHBoxLayout()
        
        self.voc_to_yolo_btn = QPushButton("Convert VOC to YOLO")
        self.voc_to_yolo_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.voc_to_yolo_btn.setStyleSheet("""
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
        self.voc_to_yolo_btn.clicked.connect(self.convert_voc_to_yolo)
        buttons_layout.addWidget(self.voc_to_yolo_btn)
        
        self.yolo_to_voc_btn = QPushButton("Convert YOLO to VOC")
        self.yolo_to_voc_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.yolo_to_voc_btn.setStyleSheet("""
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
        self.yolo_to_voc_btn.clicked.connect(self.convert_yolo_to_voc)
        buttons_layout.addWidget(self.yolo_to_voc_btn)
        
        self.yolo_to_flat_btn = QPushButton("Convert YOLO to Flat")
        self.yolo_to_flat_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.yolo_to_flat_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:pressed {
                background-color: #D84315;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.yolo_to_flat_btn.clicked.connect(self.convert_yolo_to_flat)
        buttons_layout.addWidget(self.yolo_to_flat_btn)
        
        layout.addLayout(buttons_layout)
        
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
    
    def browse_yolo_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select YOLO Dataset Directory")
        if directory:
            self.yolo_dataset_edit.setText(directory)
    
    def browse_yolo_classes_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Classes File", "",
            "Text files (*.names *.txt);;All files (*)"
        )
        if file_path:
            self.yolo_classes_file_edit.setText(file_path)
    
    def browse_flat_input(self):
        directory = QFileDialog.getExistingDirectory(self, "Select YOLO Dataset Directory")
        if directory:
            self.flat_input_edit.setText(directory)
    
    def browse_flat_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.flat_output_edit.setText(directory)
    
    def convert_voc_to_yolo(self):
        xml_dir = self.xml_dir_edit.text().strip()
        classes_file = self.classes_file_edit.text().strip()
        
        if not xml_dir:
            self.parent.show_error("Please specify XML directory")
            return
        
        try:
            self.voc_to_yolo_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_voc_conversion, xml_dir, classes_file)
            self.worker.finished.connect(self.on_voc_conversion_complete)
            self.worker.error.connect(self.on_voc_conversion_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Conversion failed: {e}")
    
    def _do_voc_conversion(self, xml_dir, classes_file):
        import glob
        import xml.etree.ElementTree as ET
        import os
        import shutil
        from pathlib import Path
        
        # determine if this is a full VOC dataset or just annotations directory
        xml_path = Path(xml_dir)
        is_full_voc = False
        voc_root = xml_path
        images_dir = xml_path  # default to same directory
        
        # check if this is standard VOC structure
        if xml_path.name == "Annotations":
            voc_root = xml_path.parent
            if (voc_root / "JPEGImages").exists():
                images_dir = voc_root / "JPEGImages"
                is_full_voc = True
            elif (voc_root / "images").exists():
                images_dir = voc_root / "images"
                is_full_voc = True
        else:
            # check if images directory exists in same parent
            possible_img_dirs = ["JPEGImages", "images", "Images"]
            for img_dir_name in possible_img_dirs:
                candidate = xml_path.parent / img_dir_name
                if candidate.exists():
                    images_dir = candidate
                    is_full_voc = True
                    break
        
        print(f"processing XML directory: {xml_path}")
        print(f"processing images directory: {images_dir}")
        
        # Load or auto-detect classes
        classes_dict = {}
        auto_detected_classes = set()
        
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                for idx, line in enumerate(f.readlines()):
                    class_name = line.strip()
                    if class_name:
                        classes_dict[class_name] = idx
            print(f"loaded {len(classes_dict)} classes from file: {classes_file}")
        else:
            print("auto-detecting classes from XML files...")
            # auto-detect classes from XML files
            xml_files = list(Path(xml_dir).glob('*.xml'))
            print(f"found {len(xml_files)} XML files for class detection")
            
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    for obj in tree.findall('object'):
                        name_elem = obj.find('name')
                        if name_elem is not None and name_elem.text:
                            class_name = name_elem.text.strip()
                            auto_detected_classes.add(class_name)
                except Exception as e:
                    print(f"warning: failed to parse {xml_file}: {e}")
                    continue
            
            # create classes dictionary
            for idx, class_name in enumerate(sorted(auto_detected_classes)):
                classes_dict[class_name] = idx
                
            print(f"auto-detected {len(classes_dict)} classes: {', '.join(sorted(auto_detected_classes))}")
        
        if not classes_dict:
            raise Exception("no classes found. either provide a classes file or ensure XML files contain class names")
        
        # create output directory structure - directly output to images/labels
        if is_full_voc:
            output_root = voc_root / "images"
            labels_root = voc_root / "labels"
        else:
            output_root = xml_path.parent / "images"
            labels_root = xml_path.parent / "labels"
        
        # create directories
        output_root.mkdir(parents=True, exist_ok=True)
        labels_root.mkdir(parents=True, exist_ok=True)
        
        # get all XML files
        xml_files = list(Path(xml_dir).glob('*.xml'))
        print(f"processing {len(xml_files)} XML files...")
        
        converted_count = 0
        copied_images = 0
        skipped_files = []
        error_files = []
        
        for i, xml_file in enumerate(xml_files):
            if i % 1000 == 0:
                print(f"processed {i}/{len(xml_files)} files...")
                
            try:
                # parse XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # get image dimensions
                size = root.find('size')
                if size is None:
                    error_files.append(f"{xml_file.name} (no size element)")
                    continue
                    
                width_elem = size.find('width')
                height_elem = size.find('height')
                
                if width_elem is None or height_elem is None:
                    error_files.append(f"{xml_file.name} (missing width/height)")
                    continue
                    
                try:
                    width = int(width_elem.text)
                    height = int(height_elem.text)
                except (ValueError, TypeError):
                    error_files.append(f"{xml_file.name} (invalid width/height values)")
                    continue
                
                # STANDARD PASCAL VOC APPROACH: use filename from XML content
                filename_elem = root.find('filename')
                if filename_elem is None or not filename_elem.text:
                    error_files.append(f"{xml_file.name} (missing filename element)")
                    continue
                
                image_filename = filename_elem.text.strip()
                image_path = images_dir / image_filename
                
                # verify image file exists exactly as specified in XML
                if not image_path.exists():
                    skipped_files.append(f"{xml_file.name} (image file not found: {image_filename})")
                    continue
                
                # process annotations
                lines = []
                objects = root.findall('object')
                
                for obj in objects:
                    name_elem = obj.find('name')
                    if name_elem is None or not name_elem.text:
                        continue
                        
                    class_name = name_elem.text.strip()
                    if class_name not in classes_dict:
                        print(f"warning: unknown class '{class_name}' in {xml_file.name}, skipping object")
                        continue
                        
                    label = classes_dict[class_name]
                    bbox = obj.find('bndbox')
                    
                    if bbox is None:
                        continue
                        
                    try:
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                    except (AttributeError, ValueError, TypeError):
                        print(f"warning: invalid bbox in {xml_file.name}, skipping object")
                        continue
                    
                    # convert to YOLO format (normalized center coordinates and dimensions)
                    cx = (xmax + xmin) * 0.5 / width
                    cy = (ymax + ymin) * 0.5 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                    
                    # clamp values to [0, 1] to handle any rounding errors
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    
                    # skip invalid boxes (width or height is 0)
                    if w <= 0 or h <= 0:
                        print(f"warning: invalid box dimensions in {xml_file.name}, skipping object")
                        continue
                    
                    line = f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    lines.append(line)
                
                # save YOLO format label file
                label_filename = image_path.stem + ".txt"
                label_path = labels_root / label_filename
                
                with open(label_path, "w", encoding='utf-8') as f:
                    f.writelines(lines)
                
                # copy image to output directory
                dest_image_path = output_root / image_path.name
                if not dest_image_path.exists():
                    shutil.copy2(image_path, dest_image_path)
                    copied_images += 1
                
                converted_count += 1
                
            except Exception as e:
                error_files.append(f"{xml_file.name} (error: {str(e)})")
                print(f"error processing {xml_file.name}: {e}")
        
        print(f"conversion completed. processed {converted_count}/{len(xml_files)} files")
        
        # create classes.txt file in parent directory
        if is_full_voc:
            classes_file_path = voc_root / "classes.txt"
            data_yaml_path = voc_root / "data.yaml"
        else:
            classes_file_path = xml_path.parent / "classes.txt"
            data_yaml_path = xml_path.parent / "data.yaml"
            
        with open(classes_file_path, 'w', encoding='utf-8') as f:
            for class_name in sorted(classes_dict.keys(), key=lambda x: classes_dict[x]):
                f.write(f"{class_name}\n")
        
        # create data.yaml file
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {classes_file_path.parent.absolute()}\n")
            f.write("train: images\n")
            f.write("val: images\n")
            f.write(f"nc: {len(classes_dict)}\n")
            f.write("names:\n")
            for idx, class_name in sorted([(v, k) for k, v in classes_dict.items()]):
                f.write(f"  {idx}: {class_name}\n")
        
        # create summary
        summary_lines = [
            f"VOC to YOLO conversion completed:",
            f"- processed {len(xml_files)} XML files",
            f"- successfully converted {converted_count} files",
            f"- copied {copied_images} unique images",
            f"- found {len(classes_dict)} classes: {', '.join(sorted(classes_dict.keys()))}",
            f"- output structure: images/ and labels/ directories",
            f"- created files: classes.txt, data.yaml"
        ]
        
        if skipped_files:
            summary_lines.append(f"- skipped {len(skipped_files)} files (missing images)")
            if len(skipped_files) <= 10:
                for skipped in skipped_files:
                    summary_lines.append(f"  • {skipped}")
            else:
                for skipped in skipped_files[:5]:
                    summary_lines.append(f"  • {skipped}")
                summary_lines.append(f"  • ... and {len(skipped_files) - 5} more")
        
        if error_files:
            summary_lines.append(f"- {len(error_files)} files had errors")
            if len(error_files) <= 10:
                for error in error_files:
                    summary_lines.append(f"  • {error}")
            else:
                for error in error_files[:5]:
                    summary_lines.append(f"  • {error}")
                summary_lines.append(f"  • ... and {len(error_files) - 5} more")
        
        if not classes_file and auto_detected_classes:
            summary_lines.append(f"- auto-detected classes: {', '.join(sorted(auto_detected_classes))}")
        
        # final validation
        total_issues = len(skipped_files) + len(error_files)
        if total_issues == 0:
            summary_lines.append("✓ all files processed successfully!")
        elif converted_count > 0:
            success_rate = (converted_count / len(xml_files)) * 100
            summary_lines.append(f"success rate: {success_rate:.1f}%")
        
        return "\n".join(summary_lines)
    
    def convert_yolo_to_voc(self):
        yolo_dataset = self.yolo_dataset_edit.text().strip()
        classes_file = self.yolo_classes_file_edit.text().strip()
        
        if not yolo_dataset:
            self.parent.show_error("Please specify YOLO dataset directory")
            return
            
        if not classes_file:
            self.parent.show_error("Please specify classes file for YOLO to VOC conversion")
            return
        
        try:
            self.yolo_to_voc_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_yolo_conversion, yolo_dataset, classes_file)
            self.worker.finished.connect(self.on_yolo_conversion_complete)
            self.worker.error.connect(self.on_yolo_conversion_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Conversion failed: {e}")
    
    def _do_yolo_conversion(self, yolo_dataset, classes_file):
        import os
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from PIL import Image
        import xml.dom.minidom
        
        dataset_path = Path(yolo_dataset)
        
        # find images and labels directories
        images_dir = None
        labels_dir = None
        
        # check for standard YOLO structure
        if (dataset_path / "images").exists():
            images_dir = dataset_path / "images"
        elif (dataset_path / "train" / "images").exists():
            images_dir = dataset_path / "train" / "images"
        else:
            # look for any directory containing images
            for subdir in dataset_path.iterdir():
                if subdir.is_dir():
                    image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
                    if image_files:
                        images_dir = subdir
                        break
        
        if (dataset_path / "labels").exists():
            labels_dir = dataset_path / "labels"
        elif (dataset_path / "train" / "labels").exists():
            labels_dir = dataset_path / "train" / "labels"
        else:
            # look for directory containing .txt files
            for subdir in dataset_path.iterdir():
                if subdir.is_dir():
                    txt_files = list(subdir.glob("*.txt"))
                    if txt_files:
                        labels_dir = subdir
                        break
        
        if not images_dir or not images_dir.exists():
            raise Exception(f"images directory not found in {yolo_dataset}")
        
        if not labels_dir or not labels_dir.exists():
            raise Exception(f"labels directory not found in {yolo_dataset}")
        
        print(f"processing YOLO dataset: {dataset_path}")
        print(f"images directory: {images_dir}")
        print(f"labels directory: {labels_dir}")
        
        # load classes
        classes = []
        if not os.path.exists(classes_file):
            raise Exception(f"classes file not found: {classes_file}")
            
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                class_name = line.strip()
                if class_name:
                    classes.append(class_name)
        
        print(f"loaded {len(classes)} classes: {', '.join(classes)}")
        
        if not classes:
            raise Exception("no classes found in classes file")
        
        # create output directories
        output_root = dataset_path / "VOC_Dataset"
        voc_annotations_dir = output_root / "Annotations"
        voc_images_dir = output_root / "JPEGImages"
        voc_imagesets_dir = output_root / "ImageSets" / "Main"
        
        voc_annotations_dir.mkdir(parents=True, exist_ok=True)
        voc_images_dir.mkdir(parents=True, exist_ok=True)
        voc_imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        print(f"found {len(image_files)} image files")
        
        converted_count = 0
        copied_images = 0
        skipped_files = []
        error_files = []
        all_image_names = []
        
        for i, image_file in enumerate(image_files):
            if i % 1000 == 0:
                print(f"processed {i}/{len(image_files)} files...")
            
            try:
                # get corresponding label file
                label_file = labels_dir / (image_file.stem + ".txt")
                
                if not label_file.exists():
                    # create empty XML for images without labels
                    try:
                        with Image.open(image_file) as img:
                            width, height = img.size
                        
                        # create empty XML
                        xml_content = self._create_voc_xml(
                            image_file.name, width, height, [], classes
                        )
                        
                        xml_file = voc_annotations_dir / (image_file.stem + ".xml")
                        with open(xml_file, 'w', encoding='utf-8') as f:
                            f.write(xml_content)
                        
                        # copy image
                        dest_image = voc_images_dir / image_file.name
                        if not dest_image.exists():
                            import shutil
                            shutil.copy2(image_file, dest_image)
                            copied_images += 1
                        
                        all_image_names.append(image_file.stem)
                        converted_count += 1
                        continue
                        
                    except Exception as e:
                        error_files.append(f"{image_file.name} (no label, failed to get image size: {e})")
                        continue
                
                # read image dimensions
                try:
                    with Image.open(image_file) as img:
                        width, height = img.size
                except Exception as e:
                    error_files.append(f"{image_file.name} (failed to read image: {e})")
                    continue
                
                # read YOLO annotations
                objects = []
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) != 5:
                                print(f"warning: invalid annotation format in {label_file.name}, line {line_num}")
                                continue
                            
                            try:
                                class_id = int(parts[0])
                                if class_id >= len(classes) or class_id < 0:
                                    print(f"warning: invalid class ID {class_id} in {label_file.name}")
                                    continue
                                
                                cx, cy, w, h = map(float, parts[1:5])
                                
                                # convert from YOLO format to VOC format
                                # YOLO: normalized center coordinates and dimensions
                                # VOC: absolute pixel coordinates (xmin, ymin, xmax, ymax)
                                xmin = int((cx - w/2) * width)
                                ymin = int((cy - h/2) * height)
                                xmax = int((cx + w/2) * width)
                                ymax = int((cy + h/2) * height)
                                
                                # clamp to image boundaries
                                xmin = max(0, min(width-1, xmin))
                                ymin = max(0, min(height-1, ymin))
                                xmax = max(1, min(width, xmax))
                                ymax = max(1, min(height, ymax))
                                
                                # skip invalid boxes
                                if xmax <= xmin or ymax <= ymin:
                                    print(f"warning: invalid box dimensions in {label_file.name}, line {line_num}")
                                    continue
                                
                                objects.append({
                                    'class': classes[class_id],
                                    'xmin': xmin,
                                    'ymin': ymin,
                                    'xmax': xmax,
                                    'ymax': ymax
                                })
                                
                            except (ValueError, IndexError) as e:
                                print(f"warning: invalid annotation in {label_file.name}, line {line_num}: {e}")
                                continue
                                
                except Exception as e:
                    error_files.append(f"{image_file.name} (failed to read labels: {e})")
                    continue
                
                # create VOC XML
                xml_content = self._create_voc_xml(
                    image_file.name, width, height, objects, classes
                )
                
                # save XML file
                xml_file = voc_annotations_dir / (image_file.stem + ".xml")
                with open(xml_file, 'w', encoding='utf-8') as f:
                    f.write(xml_content)
                
                # copy image
                dest_image = voc_images_dir / image_file.name
                if not dest_image.exists():
                    import shutil
                    shutil.copy2(image_file, dest_image)
                    copied_images += 1
                
                all_image_names.append(image_file.stem)
                converted_count += 1
                
            except Exception as e:
                error_files.append(f"{image_file.name} (error: {str(e)})")
                print(f"error processing {image_file.name}: {e}")
        
        print(f"conversion completed. processed {converted_count}/{len(image_files)} files")
        
        # create ImageSets files
        train_file = voc_imagesets_dir / "train.txt"
        val_file = voc_imagesets_dir / "val.txt"
        trainval_file = voc_imagesets_dir / "trainval.txt"
        
        with open(trainval_file, 'w', encoding='utf-8') as f:
            for name in sorted(all_image_names):
                f.write(f"{name}\n")
        
        # split 80/20 for train/val
        import random
        random.seed(42)
        shuffled_names = all_image_names.copy()
        random.shuffle(shuffled_names)
        
        split_idx = int(len(shuffled_names) * 0.8)
        train_names = shuffled_names[:split_idx]
        val_names = shuffled_names[split_idx:]
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for name in sorted(train_names):
                f.write(f"{name}\n")
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for name in sorted(val_names):
                f.write(f"{name}\n")
        
        # create summary
        summary_lines = [
            f"YOLO to VOC conversion completed:",
            f"- processed {len(image_files)} image files",
            f"- successfully converted {converted_count} files",
            f"- copied {copied_images} unique images",
            f"- used {len(classes)} classes: {', '.join(classes)}",
            f"- output directory: {output_root}",
            f"- created: Annotations/, JPEGImages/, ImageSets/",
            f"- train/val split: {len(train_names)}/{len(val_names)} files"
        ]
        
        if skipped_files:
            summary_lines.append(f"- skipped {len(skipped_files)} files")
            if len(skipped_files) <= 10:
                for skipped in skipped_files:
                    summary_lines.append(f"  • {skipped}")
            else:
                for skipped in skipped_files[:5]:
                    summary_lines.append(f"  • {skipped}")
                summary_lines.append(f"  • ... and {len(skipped_files) - 5} more")
        
        if error_files:
            summary_lines.append(f"- {len(error_files)} files had errors")
            if len(error_files) <= 10:
                for error in error_files:
                    summary_lines.append(f"  • {error}")
            else:
                for error in error_files[:5]:
                    summary_lines.append(f"  • {error}")
                summary_lines.append(f"  • ... and {len(error_files) - 5} more")
        
        # final validation
        total_issues = len(skipped_files) + len(error_files)
        if total_issues == 0:
            summary_lines.append("✓ all files processed successfully!")
        elif converted_count > 0:
            success_rate = (converted_count / len(image_files)) * 100
            summary_lines.append(f"success rate: {success_rate:.1f}%")
        
        return "\n".join(summary_lines)
    
    def _create_voc_xml(self, filename, width, height, objects, classes):
        """Create VOC format XML content"""
        import xml.etree.ElementTree as ET
        import xml.dom.minidom
        
        # create root element
        root = ET.Element('annotation')
        
        # folder
        folder = ET.SubElement(root, 'folder')
        folder.text = 'VOC_Dataset'
        
        # filename
        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = filename
        
        # path (optional)
        path = ET.SubElement(root, 'path')
        path.text = filename
        
        # source
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        
        # size
        size = ET.SubElement(root, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(height)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'
        
        # segmented
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        
        # objects
        for obj in objects:
            object_elem = ET.SubElement(root, 'object')
            
            name = ET.SubElement(object_elem, 'name')
            name.text = obj['class']
            
            pose = ET.SubElement(object_elem, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(object_elem, 'truncated')
            truncated.text = '0'
            
            difficult = ET.SubElement(object_elem, 'difficult')
            difficult.text = '0'
            
            bndbox = ET.SubElement(object_elem, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(obj['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(obj['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(obj['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(obj['ymax'])
        
        # convert to pretty XML string
        rough_string = ET.tostring(root, 'unicode')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ", encoding=None)
    
    def on_voc_conversion_complete(self, result):
        self.voc_to_yolo_btn.setEnabled(True)
        self.conversion_results.setText(result)
    
    def on_voc_conversion_error(self, error_msg):
        self.voc_to_yolo_btn.setEnabled(True)
        self.parent.show_error(f"VOC Conversion failed: {error_msg}")
    
    def on_yolo_conversion_complete(self, result):
        self.yolo_to_voc_btn.setEnabled(True)
        self.conversion_results.setText(result)
    
    def on_yolo_conversion_error(self, error_msg):
        self.yolo_to_voc_btn.setEnabled(True)
        self.parent.show_error(f"YOLO Conversion failed: {error_msg}")
    
    def convert_yolo_to_flat(self):
        input_dataset = self.flat_input_edit.text().strip()
        output_dir = self.flat_output_edit.text().strip()
        
        if not input_dataset:
            self.parent.show_error("Please specify YOLO dataset directory")
            return
            
        if not output_dir:
            self.parent.show_error("Please specify output directory")
            return
        
        # get options
        merge_splits = self.merge_splits_checkbox.isChecked()
        preserve_split_info = self.preserve_split_info_checkbox.isChecked()
        
        try:
            self.yolo_to_flat_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_yolo_to_flat_conversion, 
                                     input_dataset, output_dir, merge_splits, preserve_split_info)
            self.worker.finished.connect(self.on_flat_conversion_complete)
            self.worker.error.connect(self.on_flat_conversion_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Conversion failed: {e}")
    
    def _do_yolo_to_flat_conversion(self, input_dataset, output_dir, merge_splits, preserve_split_info):
        """
        Convert YOLO dataset to flat images+labels format.
        Directly calls the conversion function instead of subprocess.
        """
        try:
            # Import the conversion function directly
            from yolo_to_flat_converter import yolo_to_flat_conversion
            
            # Call the conversion function directly
            result = yolo_to_flat_conversion(
                input_dataset,
                output_dir,
                merge_splits=merge_splits,
                preserve_split_info=preserve_split_info
            )
            
            return result['summary']
            
        except ImportError as e:
            raise Exception(f"Failed to import yolo_to_flat_converter: {str(e)}")
        except Exception as e:
            raise Exception(f"YOLO to flat conversion failed: {str(e)}")
    
    def on_flat_conversion_complete(self, result):
        self.yolo_to_flat_btn.setEnabled(True)
        self.conversion_results.setText(result)
    
    def on_flat_conversion_error(self, error_msg):
        self.yolo_to_flat_btn.setEnabled(True)
        self.parent.show_error(f"YOLO to Flat Conversion failed: {error_msg}")

# Dataset Management Tab - dataset reduction, completion, and class replacement
class DatasetManagementTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Merge YOLO Datasets
        merge_group = QGroupBox("Merge Multiple YOLO Datasets")
        merge_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        merge_group.setStyleSheet("""
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
        merge_layout = QVBoxLayout()
        merge_layout.setSpacing(12)
        
        # dataset list with add/remove buttons
        datasets_label = QLabel("source datasets:")
        datasets_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        merge_layout.addWidget(datasets_label)
        
        # list widget for datasets
        self.datasets_list = QListWidget()
        self.datasets_list.setMaximumHeight(120)
        self.datasets_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                alternate-background-color: #f9f9f9;
            }
        """)
        merge_layout.addWidget(self.datasets_list)
        
        # buttons for managing dataset list
        list_buttons_layout = QHBoxLayout()
        
        self.add_dataset_btn = QPushButton("Add Dataset...")
        self.add_dataset_btn.setFont(QFont("Arial", 11))
        self.add_dataset_btn.clicked.connect(self.add_dataset)
        list_buttons_layout.addWidget(self.add_dataset_btn)
        
        self.remove_dataset_btn = QPushButton("Remove Selected")
        self.remove_dataset_btn.setFont(QFont("Arial", 11))
        self.remove_dataset_btn.clicked.connect(self.remove_selected_dataset)
        self.remove_dataset_btn.setEnabled(False)
        list_buttons_layout.addWidget(self.remove_dataset_btn)
        
        self.clear_datasets_btn = QPushButton("Clear All")
        self.clear_datasets_btn.setFont(QFont("Arial", 11))
        self.clear_datasets_btn.clicked.connect(self.clear_all_datasets)
        self.clear_datasets_btn.setEnabled(False)
        list_buttons_layout.addWidget(self.clear_datasets_btn)
        
        list_buttons_layout.addStretch()
        merge_layout.addLayout(list_buttons_layout)
        
        # enable/disable buttons based on selection
        self.datasets_list.itemSelectionChanged.connect(self.update_dataset_buttons)
        
        # output directory
        output_layout = QHBoxLayout()
        output_label = QLabel("output directory:")
        output_label.setFont(QFont("Arial", 11))
        output_layout.addWidget(output_label)
        
        self.merge_out_edit = QLineEdit()
        self.merge_out_edit.setFont(QFont("Arial", 11))
        self.merge_out_edit.setStyleSheet("""
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
        output_layout.addWidget(self.merge_out_edit)
        
        self.merge_out_browse = QPushButton("Browse...")
        self.merge_out_browse.setFont(QFont("Arial", 11))
        self.merge_out_browse.clicked.connect(self.browse_merge_out)
        output_layout.addWidget(self.merge_out_browse)
        
        merge_layout.addLayout(output_layout)
        
        merge_group.setLayout(merge_layout)
        layout.addWidget(merge_group)
        
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
        
        # Class Removal Tool
        class_remove_group = QGroupBox("Class ID Removal Tool")
        class_remove_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        class_remove_group.setStyleSheet("""
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
        class_remove_layout = QFormLayout()
        class_remove_layout.setVerticalSpacing(12)
        
        # Dataset directory for class removal
        self.remove_class_dataset_edit = QLineEdit()
        self.remove_class_dataset_edit.setFont(QFont("Arial", 11))
        self.remove_class_dataset_browse = QPushButton("Browse...")
        self.remove_class_dataset_browse.setFont(QFont("Arial", 11))
        self.remove_class_dataset_browse.clicked.connect(self.browse_remove_class_dataset)
        remove_class_dataset_row = QHBoxLayout()
        remove_class_dataset_row.addWidget(self.remove_class_dataset_edit)
        remove_class_dataset_row.addWidget(self.remove_class_dataset_browse)
        class_remove_layout.addRow("Dataset Directory:", remove_class_dataset_row)
        
        # Class ID to remove
        self.remove_class_id_spin = QSpinBox()
        self.remove_class_id_spin.setRange(0, 100)
        self.remove_class_id_spin.setValue(1)
        class_remove_layout.addRow("Class ID to Remove:", self.remove_class_id_spin)
        
        class_remove_group.setLayout(class_remove_layout)
        layout.addWidget(class_remove_group)
        
        # Cleanup Orphaned Labels Tool
        cleanup_group = QGroupBox("Cleanup Orphaned Labels")
        cleanup_group.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        cleanup_group.setStyleSheet("""
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
        cleanup_layout = QFormLayout()
        cleanup_layout.setVerticalSpacing(12)
        
        # Dataset directory for cleanup
        self.cleanup_dataset_edit = QLineEdit()
        self.cleanup_dataset_edit.setFont(QFont("Arial", 11))
        self.cleanup_dataset_browse = QPushButton("Browse...")
        self.cleanup_dataset_browse.setFont(QFont("Arial", 11))
        self.cleanup_dataset_browse.clicked.connect(self.browse_cleanup_dataset)
        cleanup_dataset_row = QHBoxLayout()
        cleanup_dataset_row.addWidget(self.cleanup_dataset_edit)
        cleanup_dataset_row.addWidget(self.cleanup_dataset_browse)
        cleanup_layout.addRow("Dataset Directory:", cleanup_dataset_row)
        
        cleanup_group.setLayout(cleanup_layout)
        layout.addWidget(cleanup_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        # Merge button
        self.merge_btn = QPushButton("Merge Datasets")
        self.merge_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.merge_btn.setStyleSheet("""
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
        self.merge_btn.clicked.connect(self.merge_datasets)
        buttons_layout.addWidget(self.merge_btn)
        
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
        
        # Remove class ID button
        self.remove_class_btn = QPushButton("Remove Class ID")
        self.remove_class_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.remove_class_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.remove_class_btn.clicked.connect(self.remove_class_id)
        buttons_layout.addWidget(self.remove_class_btn)
        
        # Cleanup orphaned labels button
        self.cleanup_btn = QPushButton("Cleanup Orphaned Labels")
        self.cleanup_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.cleanup_btn.setStyleSheet("""
            QPushButton {
                background-color: #E91E63;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #C2185B;
            }
            QPushButton:pressed {
                background-color: #AD1457;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.cleanup_btn.clicked.connect(self.cleanup_orphaned_labels)
        buttons_layout.addWidget(self.cleanup_btn)
        
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
    
    
    def browse_merge_out(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.merge_out_edit.setText(directory)
    
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
    
    def browse_remove_class_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory for Class Removal")
        if directory:
            self.remove_class_dataset_edit.setText(directory)
    
    def browse_cleanup_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Directory for Cleanup")
        if directory:
            self.cleanup_dataset_edit.setText(directory)
    
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
        """Create empty txt files for images without corresponding labels.
        supports both simple (images/labels) and standard YOLO (train|val|test)/(images|labels) structure
        """
        import os
        from pathlib import Path
        
        def find_dataset_structure(root_dir):
            """detect dataset structure and return available splits"""
            root = Path(root_dir)
            structure = {}
            
            # check for standard YOLO structure
            for split in ['train', 'val', 'test']:
                split_dir = root / split
                if split_dir.exists():
                    images_dir = split_dir / 'images'
                    if images_dir.exists():
                        labels_dir = split_dir / 'labels'
                        structure[split] = {'images': images_dir, 'labels': labels_dir}
            
            # fallback: check for direct images/labels structure
            if not structure:
                images_dir = root / 'images'
                if images_dir.exists():
                    labels_dir = root / 'labels'
                    structure['main'] = {'images': images_dir, 'labels': labels_dir}
            
            return structure
        
        dataset_structure = find_dataset_structure(dataset_dir)
        
        if not dataset_structure:
            raise Exception("no valid dataset structure found (expected images/ directory or train|val|test/images/)")
        
        total_created = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        split_stats = {}
        
        # process each split
        for split, dirs in dataset_structure.items():
            images_dir = dirs['images']
            labels_dir = dirs['labels']
            
            # create labels directory if it doesn't exist
            if not labels_dir.exists():
                labels_dir.mkdir(parents=True, exist_ok=True)
            
            created_count = 0
            
            # get all image files
            for filename in os.listdir(images_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    # get the base name without extension
                    base_name = os.path.splitext(filename)[0]
                    label_path = labels_dir / f"{base_name}.txt"
                    
                    # create empty label file if it doesn't exist
                    if not label_path.exists():
                        with open(label_path, 'w') as f:
                            pass  # create empty file
                        created_count += 1
            
            split_stats[split] = created_count
            total_created += created_count
        
        # create summary message
        if len(split_stats) == 1 and 'main' in split_stats:
            return f"created {total_created} empty label files for unlabeled images"
        else:
            summary_lines = [f"created {total_created} empty label files across splits:"]
            for split, count in split_stats.items():
                summary_lines.append(f"  - {split}: {count} files")
            return "\n".join(summary_lines)
    
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
    
    def merge_datasets(self):
        """merge multiple YOLO datasets using the new multi-source interface"""
        # collect all dataset paths from the list
        source_paths = []
        for i in range(self.datasets_list.count()):
            source_paths.append(self.datasets_list.item(i).text())
        
        out_dir = self.merge_out_edit.text().strip()
        
        if len(source_paths) < 2:
            self.parent.show_error("please add at least 2 datasets to merge")
            return
            
        if not out_dir:
            self.parent.show_error("please specify an output directory")
            return
            
        try:
            self.merge_btn.setEnabled(False)
            # use the new multi-source merge method
            self.worker = WorkerThread(DatasetManager.merge_yolo_datasets, source_paths, out_dir)
            self.worker.finished.connect(self.on_merge_complete)
            self.worker.error.connect(self.on_merge_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"merge failed: {e}")
    
    def on_merge_complete(self, result):
        self.merge_btn.setEnabled(True)
        self.advanced_results.setText(result)
    
    def on_merge_error(self, error_msg):
        self.merge_btn.setEnabled(True)
        self.parent.show_error(f"Merge failed: {error_msg}")
    
    def cleanup_orphaned_labels(self):
        """Remove label files that don't have corresponding image files"""
        dataset_dir = self.cleanup_dataset_edit.text().strip()
        
        if not dataset_dir:
            self.parent.show_error("Please specify dataset directory")
            return
        
        try:
            self.cleanup_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_orphaned_labels_cleanup, dataset_dir)
            self.worker.finished.connect(self.on_cleanup_complete)
            self.worker.error.connect(self.on_cleanup_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"Cleanup failed: {e}")
    
    def _do_orphaned_labels_cleanup(self, dataset_dir):
        """Remove label files that don't have corresponding image files"""
        import os
        
        labels_dir = os.path.join(dataset_dir, 'labels')
        images_dir = os.path.join(dataset_dir, 'images')
        
        if not os.path.exists(labels_dir):
            raise Exception(f"Labels directory not found: {labels_dir}")
        
        if not os.path.exists(images_dir):
            raise Exception(f"Images directory not found: {images_dir}")
        
        removed_count = 0
        total_labels = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # Get all existing image files (without extension) for quick lookup
        existing_images = set()
        for filename in os.listdir(images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                base_name = os.path.splitext(filename)[0]
                existing_images.add(base_name)
        
        # Check each label file
        for filename in os.listdir(labels_dir):
            if not filename.endswith('.txt'):
                continue
                
            total_labels += 1
            base_name = os.path.splitext(filename)[0]
            
            # If no corresponding image exists, remove the label file
            if base_name not in existing_images:
                label_path = os.path.join(labels_dir, filename)
                try:
                    os.remove(label_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {label_path}: {e}")
        
        return f"Removed {removed_count} orphaned label files out of {total_labels} total labels"
    
    def on_cleanup_complete(self, result):
        self.cleanup_btn.setEnabled(True)
        self.advanced_results.setText(result)
    
    def on_cleanup_error(self, error_msg):
        self.cleanup_btn.setEnabled(True)
        self.parent.show_error(f"Cleanup failed: {error_msg}")
    
    def remove_class_id(self):
        """Remove all label lines with the specified class ID from the dataset"""
        dataset_dir = self.remove_class_dataset_edit.text().strip()
        class_id = self.remove_class_id_spin.value()
        
        if not dataset_dir:
            self.parent.show_error("please specify dataset directory")
            return
        
        try:
            self.remove_class_btn.setEnabled(False)
            self.worker = WorkerThread(self._do_class_removal, dataset_dir, class_id)
            self.worker.finished.connect(self.on_class_removal_complete)
            self.worker.error.connect(self.on_class_removal_error)
            self.worker.start()
        except Exception as e:
            self.parent.show_error(f"class removal failed: {e}")
    
    def _do_class_removal(self, dataset_dir, class_id):
        """Remove all annotation lines with the specified class ID from YOLO label files
        Keep empty label files as background samples instead of deleting them
        """
        import os
        from pathlib import Path
        
        dataset_path = Path(dataset_dir)
        
        # find labels directory - support multiple structures
        possible_label_dirs = [
            dataset_path / "labels",
            dataset_path / "train" / "labels",
            dataset_path / "val" / "labels",
            dataset_path / "test" / "labels"
        ]
        
        found_dirs = []
        for label_dir in possible_label_dirs:
            if label_dir.exists() and any(label_dir.glob("*.txt")):
                found_dirs.append(label_dir)
        
        if not found_dirs:
            raise Exception(f"no label directories found in dataset: {dataset_dir}")
        
        total_files_processed = 0
        total_lines_removed = 0
        files_became_empty = 0
        
        for labels_dir in found_dirs:
            print(f"processing labels directory: {labels_dir}")
            
            for label_file in labels_dir.glob("*.txt"):
                try:
                    # read all lines
                    with open(label_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    original_line_count = len([line for line in lines if line.strip()])
                    
                    # filter out lines with the target class ID
                    filtered_lines = []
                    lines_removed_in_file = 0
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue  # skip empty lines
                        
                        parts = line.split()
                        if len(parts) >= 5:  # valid YOLO format
                            try:
                                line_class_id = int(parts[0])
                                if line_class_id != class_id:
                                    filtered_lines.append(line)
                                else:
                                    lines_removed_in_file += 1
                            except ValueError:
                                # keep lines with invalid class ID format
                                filtered_lines.append(line)
                        else:
                            # keep lines with invalid format
                            filtered_lines.append(line)
                    
                    # update counters
                    total_lines_removed += lines_removed_in_file
                    
                    # always write back to file (even if empty)
                    with open(label_file, 'w', encoding='utf-8') as f:
                        if filtered_lines:
                            for line in filtered_lines:
                                f.write(line + '\n')
                        # if no filtered_lines, create empty file (background sample)
                    
                    # track files that became empty (now background samples)
                    if original_line_count > 0 and len(filtered_lines) == 0:
                        files_became_empty += 1
                        print(f"file {label_file.name} became empty - now a background sample")
                    
                    total_files_processed += 1
                    
                except Exception as e:
                    print(f"error processing {label_file}: {e}")
                    continue
        
        # create summary
        summary_lines = [
            f"class ID removal completed:",
            f"- processed {total_files_processed} label files",
            f"- removed {total_lines_removed} annotation lines with class ID {class_id}",
            f"- {files_became_empty} files became empty (preserved as background samples)",
            f"- processed directories: {', '.join([str(d) for d in found_dirs])}"
        ]
        
        if total_lines_removed == 0:
            summary_lines.append(f"⚠ no annotations found with class ID {class_id}")
        else:
            summary_lines.append(f"✓ successfully removed all instances of class ID {class_id}")
            if files_became_empty > 0:
                summary_lines.append(f"✓ preserved {files_became_empty} images as background samples")
        
        return "\n".join(summary_lines)
    
    def on_class_removal_complete(self, result):
        self.remove_class_btn.setEnabled(True)
        self.advanced_results.setText(result)
    
    def on_class_removal_error(self, error_msg):
        self.remove_class_btn.setEnabled(True)
        self.parent.show_error(f"class removal failed: {error_msg}")
    
    # dataset list management methods
    def add_dataset(self):
        """add a dataset to the list for merging"""
        directory = QFileDialog.getExistingDirectory(self, "select dataset directory")
        if directory:
            # check if directory is already in the list
            for i in range(self.datasets_list.count()):
                if self.datasets_list.item(i).text() == directory:
                    self.parent.show_error("this dataset is already in the list")
                    return
            
            self.datasets_list.addItem(directory)
            self.update_dataset_buttons()
    
    def remove_selected_dataset(self):
        """remove the selected dataset from the list"""
        current_row = self.datasets_list.currentRow()
        if current_row >= 0:
            self.datasets_list.takeItem(current_row)
            self.update_dataset_buttons()
    
    def clear_all_datasets(self):
        """clear all datasets from the list"""
        self.datasets_list.clear()
        self.update_dataset_buttons()
    
    def update_dataset_buttons(self):
        """enable/disable buttons based on list state and selection"""
        has_items = self.datasets_list.count() > 0
        has_selection = self.datasets_list.currentRow() >= 0
        
        self.remove_dataset_btn.setEnabled(has_selection)
        self.clear_datasets_btn.setEnabled(has_items)

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
        self.video_processing_tab = VideoProcessingTab(self)
        
        tab_widget.addTab(self.data_processing_tab, "Data")
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
