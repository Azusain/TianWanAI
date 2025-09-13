// tauri invoke function
const { invoke } = window.__TAURI__.tauri;

// utility functions
function showResults(message, type = 'info') {
    const resultsContent = document.getElementById('results-content');
    const timestamp = new Date().toLocaleTimeString();
    const className = type === 'success' ? 'success' : type === 'error' ? 'error' : 'info';
    
    // remove placeholder if present
    const placeholder = resultsContent.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    const resultDiv = document.createElement('div');
    resultDiv.className = className;
    resultDiv.innerHTML = `[${timestamp}] ${message}\n\n`;
    resultsContent.appendChild(resultDiv);
    resultsContent.scrollTop = resultsContent.scrollHeight;
}

function setLoading(button, loading = true) {
    if (loading) {
        button.disabled = true;
        button.dataset.originalText = button.textContent;
        button.textContent = 'processing...';
        button.classList.add('loading');
    } else {
        button.disabled = false;
        button.textContent = button.dataset.originalText || button.textContent.replace('processing...', '');
        button.classList.remove('loading');
    }
}

async function selectDirectory(title = 'select directory') {
    try {
        const result = await invoke('select_directory', { title });
        return result;
    } catch (error) {
        console.error('error selecting directory:', error);
        showResults(`error selecting directory: ${error}`, 'error');
        return null;
    }
}

async function selectFile(title = 'select file', filters = null) {
    try {
        const result = await invoke('select_file', { title, filters });
        return result;
    } catch (error) {
        console.error('error selecting file:', error);
        showResults(`error selecting file: ${error}`, 'error');
        return null;
    }
}

// tab system
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const toolPanels = document.querySelectorAll('.tool-panel');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            
            // update tab buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // update panels
            toolPanels.forEach(panel => panel.classList.remove('active'));
            const targetPanel = document.getElementById(`${tabName}-panel`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}

// dataset management functions
async function analyzeDataset() {
    const datasetPath = document.getElementById('dataset-path').value;
    const imagesDir = document.getElementById('images-dir').value || null;
    const labelsDir = document.getElementById('labels-dir').value || null;
    
    if (!datasetPath) {
        showResults('please select a dataset path first', 'error');
        return;
    }
    
    const button = document.getElementById('analyze-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('analyze_dataset', {
            request: {
                dataset_path: datasetPath,
                images_dir: imagesDir,
                labels_dir: labelsDir
            }
        });
        
        if (result.success) {
            showResults(`dataset analysis completed successfully:\n${JSON.stringify(result.data, null, 2)}`, 'success');
        } else {
            showResults(`dataset analysis failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`dataset analysis error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

async function splitDataset() {
    const datasetPath = document.getElementById('dataset-path').value;
    const outputPath = document.getElementById('split-output-path').value;
    const trainRatio = parseFloat(document.getElementById('train-ratio').value);
    const splitMode = document.getElementById('split-mode').value;
    const imagesDir = document.getElementById('images-dir').value || null;
    const labelsDir = document.getElementById('labels-dir').value || null;
    
    if (!datasetPath || !outputPath) {
        showResults('please select dataset path and output path', 'error');
        return;
    }
    
    const button = document.getElementById('split-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('split_dataset', {
            request: {
                dataset_path: datasetPath,
                output_path: outputPath,
                train_ratio: trainRatio,
                split_mode: splitMode,
                seed: 42,
                images_dir: imagesDir,
                labels_dir: labelsDir
            }
        });
        
        if (result.success) {
            showResults(`dataset split completed:\n${result.data}`, 'success');
        } else {
            showResults(`dataset split failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`dataset split error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

async function visualizeSamples() {
    const datasetPath = document.getElementById('dataset-path').value;
    const sampleCount = parseInt(document.getElementById('sample-count').value);
    const outputDir = document.getElementById('viz-output-dir').value || null;
    const imagesDir = document.getElementById('images-dir').value || null;
    const labelsDir = document.getElementById('labels-dir').value || null;
    
    if (!datasetPath) {
        showResults('please select a dataset path first', 'error');
        return;
    }
    
    const button = document.getElementById('visualize-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('visualize_samples', {
            request: {
                dataset_path: datasetPath,
                output_dir: outputDir,
                sample_count: sampleCount,
                images_dir: imagesDir,
                labels_dir: labelsDir
            }
        });
        
        if (result.success) {
            showResults(`sample visualization completed:\ngenerated files: ${result.data.join(', ')}`, 'success');
        } else {
            showResults(`sample visualization failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`sample visualization error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

// format conversion functions
async function convertVocToYolo() {
    const vocDir = document.getElementById('voc-dir').value;
    const classesFile = document.getElementById('classes-file').value;
    const outputDir = document.getElementById('conversion-output-dir').value;
    
    if (!vocDir || !classesFile || !outputDir) {
        showResults('please fill all required fields for conversion', 'error');
        return;
    }
    
    const button = document.getElementById('convert-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('convert_voc_to_yolo', {
            request: {
                voc_dir: vocDir,
                output_dir: outputDir,
                classes_file: classesFile
            }
        });
        
        if (result.success) {
            showResults(`voc to yolo conversion completed:\n${result.data}`, 'success');
        } else {
            showResults(`voc to yolo conversion failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`voc to yolo conversion error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

async function batchConvertVocToYolo() {
    const vocDirsText = document.getElementById('batch-voc-dirs').value;
    const outputBase = document.getElementById('batch-output-base').value;
    const classesFile = document.getElementById('classes-file').value;
    
    if (!vocDirsText || !outputBase || !classesFile) {
        showResults('please fill all required fields for batch conversion', 'error');
        return;
    }
    
    const vocDirs = vocDirsText.split('\n').filter(line => line.trim().length > 0);
    if (vocDirs.length === 0) {
        showResults('please provide at least one voc directory', 'error');
        return;
    }
    
    const button = document.getElementById('batch-convert-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('batch_convert_voc_to_yolo', {
            vocDirs,
            outputBase,
            classesFile
        });
        
        if (result.success) {
            showResults(`batch conversion completed:\n${result.data}`, 'success');
        } else {
            showResults(`batch conversion failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`batch conversion error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

// video processing functions
async function extractVideoFrames() {
    const videoFile = document.getElementById('video-file').value;
    const outputDir = document.getElementById('video-output-dir').value;
    const prefix = document.getElementById('frame-prefix').value;
    const targetFrames = document.getElementById('target-frames').value ? parseInt(document.getElementById('target-frames').value) : null;
    const interval = document.getElementById('frame-interval').value ? parseInt(document.getElementById('frame-interval').value) : null;
    const outputFormat = document.getElementById('output-format').value;
    const quality = parseInt(document.getElementById('output-quality').value);
    
    if (!videoFile || !outputDir) {
        showResults('please select video file and output directory', 'error');
        return;
    }
    
    const button = document.getElementById('extract-frames-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('extract_video_frames', {
            request: {
                video_path: videoFile,
                output_folder: outputDir,
                prefix: prefix,
                target_frames: targetFrames,
                interval: interval,
                output_format: outputFormat,
                quality: quality
            }
        });
        
        if (result.success) {
            showResults(`frame extraction completed:\n${result.data}`, 'success');
        } else {
            showResults(`frame extraction failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`frame extraction error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

async function batchExtractVideoFrames() {
    const inputDir = document.getElementById('batch-video-input').value;
    const outputBase = document.getElementById('batch-video-output').value;
    const prefix = document.getElementById('frame-prefix').value;
    const targetFrames = document.getElementById('target-frames').value ? parseInt(document.getElementById('target-frames').value) : null;
    const interval = document.getElementById('frame-interval').value ? parseInt(document.getElementById('frame-interval').value) : null;
    const outputFormat = document.getElementById('output-format').value;
    const quality = parseInt(document.getElementById('output-quality').value);
    
    if (!inputDir || !outputBase) {
        showResults('please select input directory and output base directory', 'error');
        return;
    }
    
    const button = document.getElementById('batch-extract-btn');
    setLoading(button, true);
    
    try {
        const result = await invoke('batch_extract_video_frames', {
            request: {
                input_dir: inputDir,
                output_base: outputBase,
                target_frames: targetFrames,
                interval: interval,
                prefix: prefix,
                output_format: outputFormat,
                quality: quality
            }
        });
        
        if (result.success) {
            showResults(`batch frame extraction completed:\n${result.data}`, 'success');
        } else {
            showResults(`batch frame extraction failed: ${result.message}`, 'error');
        }
    } catch (error) {
        showResults(`batch frame extraction error: ${error}`, 'error');
    } finally {
        setLoading(button, false);
    }
}

// initialize the application
function initializeApp() {
    // initialize tabs
    initializeTabs();
    
    // dataset management event listeners
    document.getElementById('select-dataset-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select dataset directory');
        if (path) {
            document.getElementById('dataset-path').value = path;
        }
    });
    
    document.getElementById('select-split-output-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select split output directory');
        if (path) {
            document.getElementById('split-output-path').value = path;
        }
    });
    
    document.getElementById('select-viz-output-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select visualization output directory');
        if (path) {
            document.getElementById('viz-output-dir').value = path;
        }
    });
    
    document.getElementById('analyze-btn').addEventListener('click', analyzeDataset);
    document.getElementById('split-btn').addEventListener('click', splitDataset);
    document.getElementById('visualize-btn').addEventListener('click', visualizeSamples);
    
    // format conversion event listeners
    document.getElementById('select-voc-dir-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select voc directory');
        if (path) {
            document.getElementById('voc-dir').value = path;
        }
    });
    
    document.getElementById('select-classes-file-btn').addEventListener('click', async () => {
        const path = await selectFile('select classes file', [['text files', ['txt']]]);
        if (path) {
            document.getElementById('classes-file').value = path;
        }
    });
    
    document.getElementById('select-conversion-output-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select conversion output directory');
        if (path) {
            document.getElementById('conversion-output-dir').value = path;
        }
    });
    
    document.getElementById('select-batch-output-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select batch output base directory');
        if (path) {
            document.getElementById('batch-output-base').value = path;
        }
    });
    
    document.getElementById('convert-btn').addEventListener('click', convertVocToYolo);
    document.getElementById('batch-convert-btn').addEventListener('click', batchConvertVocToYolo);
    
    // video processing event listeners
    document.getElementById('select-video-file-btn').addEventListener('click', async () => {
        const path = await selectFile('select video file', [['video files', ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']]]);
        if (path) {
            document.getElementById('video-file').value = path;
        }
    });
    
    document.getElementById('select-video-output-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select video output directory');
        if (path) {
            document.getElementById('video-output-dir').value = path;
        }
    });
    
    document.getElementById('select-batch-video-input-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select batch video input directory');
        if (path) {
            document.getElementById('batch-video-input').value = path;
        }
    });
    
    document.getElementById('select-batch-video-output-btn').addEventListener('click', async () => {
        const path = await selectDirectory('select batch video output directory');
        if (path) {
            document.getElementById('batch-video-output').value = path;
        }
    });
    
    document.getElementById('extract-frames-btn').addEventListener('click', extractVideoFrames);
    document.getElementById('batch-extract-btn').addEventListener('click', batchExtractVideoFrames);
    
    // results panel event listener
    document.getElementById('clear-results-btn').addEventListener('click', () => {
        const resultsContent = document.getElementById('results-content');
        resultsContent.innerHTML = '<p class="placeholder">results will appear here...</p>';
    });
    
    // show welcome message
    showResults('welcome to unified tool app! select a tool from the tabs above to get started.', 'info');
}

// start the application when the page loads
document.addEventListener('DOMContentLoaded', initializeApp);
