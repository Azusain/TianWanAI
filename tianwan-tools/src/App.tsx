import { useState } from "react";
// import { invoke } from "@tauri-apps/api/core";
// import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

// Mock functions for development
const invoke = async (command: string, args?: any) => {
  console.log(`[MOCK] invoke ${command}:`, args);
  if (command === "extract_frames") {
    return `Mock: would extract frames with config: ${JSON.stringify(args)}`;
  }
  if (command === "split_dataset") {
    return `Mock: would split dataset at ${args.dataset_path} with ratio ${args.train_ratio}`;
  }
  return `Mock response for ${command}`;
};

const open = async (options: any) => {
  console.log(`[MOCK] file dialog:`, options);
  if (options.directory) {
    return "C:\\mock\\selected\\directory";
  }
  return "C:\\mock\\selected\\file.mp4";
};

interface FrameExtractionConfig {
  video_path: string;
  output_dir: string;
  interval: number;
  prefix: string;
  start_idx: number;
  max_idx?: number;
  format: string;
}

function App() {
  const [activeTab, setActiveTab] = useState("frame-extraction");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  
  // Frame Extraction State
  const [videoPath, setVideoPath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [interval, setInterval] = useState(4);
  const [prefix, setPrefix] = useState("frame");
  const [startIdx, setStartIdx] = useState(1);
  const [maxIdx, setMaxIdx] = useState<number | undefined>(undefined);
  const [format, setFormat] = useState("jpg");
  
  // Dataset Splitting State
  const [datasetPath, setDatasetPath] = useState("");
  const [trainRatio, setTrainRatio] = useState(0.8);

  async function selectVideoFile() {
    try {
      const selected = await open({
        multiple: false,
        filters: [{
          name: 'Video',
          extensions: ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv']
        }]
      });
      if (selected) {
        setVideoPath(selected as string);
      }
    } catch (error) {
      console.error('error selecting video file:', error);
    }
  }

  async function selectOutputDirectory() {
    try {
      const selected = await open({
        directory: true
      });
      if (selected) {
        setOutputDir(selected as string);
      }
    } catch (error) {
      console.error('error selecting output directory:', error);
    }
  }

  async function selectDatasetDirectory() {
    try {
      const selected = await open({
        directory: true
      });
      if (selected) {
        setDatasetPath(selected as string);
      }
    } catch (error) {
      console.error('error selecting dataset directory:', error);
    }
  }

  async function extractFrames() {
    if (!videoPath || !outputDir) {
      setStatus('please select video file and output directory');
      return;
    }

    setLoading(true);
    setStatus('extracting frames...');
    
    try {
      const config: FrameExtractionConfig = {
        video_path: videoPath,
        output_dir: outputDir,
        interval,
        prefix,
        start_idx: startIdx,
        max_idx: maxIdx,
        format
      };
      
      const result = await invoke("extract_frames", { config });
      setStatus(`success: ${result}`);
    } catch (error) {
      setStatus(`error: ${error}`);
    } finally {
      setLoading(false);
    }
  }

  async function splitDataset() {
    if (!datasetPath) {
      setStatus('please select dataset directory');
      return;
    }

    setLoading(true);
    setStatus('splitting dataset...');
    
    try {
      const result = await invoke("split_dataset", { 
        dataset_path: datasetPath,
        train_ratio: trainRatio
      });
      setStatus(`success: ${result}`);
    } catch (error) {
      setStatus(`error: ${error}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🛠️ Tianwan AI Tools</h1>
        <p>computer vision and machine learning utilities</p>
      </header>

      <nav className="tab-nav">
        <button 
          className={activeTab === "frame-extraction" ? "tab active" : "tab"}
          onClick={() => setActiveTab("frame-extraction")}
        >
          🎬 frame extraction
        </button>
        <button 
          className={activeTab === "dataset-splitting" ? "tab active" : "tab"}
          onClick={() => setActiveTab("dataset-splitting")}
        >
          📊 dataset splitting
        </button>
        <button 
          className={activeTab === "dataset-stats" ? "tab active" : "tab"}
          onClick={() => setActiveTab("dataset-stats")}
        >
          📈 dataset statistics
        </button>
        <button 
          className={activeTab === "visualization" ? "tab active" : "tab"}
          onClick={() => setActiveTab("visualization")}
        >
          👁️ visualization
        </button>
      </nav>

      <main className="main-content">
        {activeTab === "frame-extraction" && (
          <div className="tool-panel">
            <h2>video frame extraction</h2>
            <div className="form-group">
              <label>video file:</label>
              <div className="file-input">
                <input 
                  type="text" 
                  value={videoPath} 
                  placeholder="select video file..."
                  readOnly
                />
                <button onClick={selectVideoFile}>browse</button>
              </div>
            </div>
            
            <div className="form-group">
              <label>output directory:</label>
              <div className="file-input">
                <input 
                  type="text" 
                  value={outputDir} 
                  placeholder="select output directory..."
                  readOnly
                />
                <button onClick={selectOutputDirectory}>browse</button>
              </div>
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label>interval:</label>
                <input 
                  type="number" 
                  value={interval} 
                  onChange={(e) => setInterval(Number(e.target.value))}
                  min="1"
                />
              </div>
              
              <div className="form-group">
                <label>prefix:</label>
                <input 
                  type="text" 
                  value={prefix} 
                  onChange={(e) => setPrefix(e.target.value)}
                />
              </div>
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label>start index:</label>
                <input 
                  type="number" 
                  value={startIdx} 
                  onChange={(e) => setStartIdx(Number(e.target.value))}
                  min="0"
                />
              </div>
              
              <div className="form-group">
                <label>max frames (optional):</label>
                <input 
                  type="number" 
                  value={maxIdx || ''} 
                  onChange={(e) => setMaxIdx(e.target.value ? Number(e.target.value) : undefined)}
                  min="1"
                  placeholder="unlimited"
                />
              </div>
            </div>
            
            <div className="form-group">
              <label>format:</label>
              <select value={format} onChange={(e) => setFormat(e.target.value)}>
                <option value="jpg">JPG (lossy, smaller)</option>
                <option value="png">PNG (lossless, larger)</option>
              </select>
            </div>
            
            <button 
              className="action-button"
              onClick={extractFrames}
              disabled={loading}
            >
              {loading ? '⏳ extracting...' : '🎬 extract frames'}
            </button>
          </div>
        )}

        {activeTab === "dataset-splitting" && (
          <div className="tool-panel">
            <h2>dataset train/validation splitting</h2>
            <div className="form-group">
              <label>dataset directory:</label>
              <div className="file-input">
                <input 
                  type="text" 
                  value={datasetPath} 
                  placeholder="select dataset directory..."
                  readOnly
                />
                <button onClick={selectDatasetDirectory}>browse</button>
              </div>
            </div>
            
            <div className="form-group">
              <label>train ratio: {trainRatio}</label>
              <input 
                type="range" 
                value={trainRatio} 
                onChange={(e) => setTrainRatio(Number(e.target.value))}
                min="0.1"
                max="0.9"
                step="0.05"
                className="slider"
              />
              <div className="ratio-info">
                <span>train: {Math.round(trainRatio * 100)}%</span>
                <span>validation: {Math.round((1 - trainRatio) * 100)}%</span>
              </div>
            </div>
            
            <button 
              className="action-button"
              onClick={splitDataset}
              disabled={loading}
            >
              {loading ? '⏳ splitting...' : '📊 split dataset'}
            </button>
          </div>
        )}

        {activeTab === "dataset-stats" && (
          <div className="tool-panel">
            <h2>dataset statistics</h2>
            <p>coming soon: analyze YOLO format datasets</p>
          </div>
        )}

        {activeTab === "visualization" && (
          <div className="tool-panel">
            <h2>dataset visualization</h2>
            <p>coming soon: visualize images with bounding boxes</p>
          </div>
        )}
      </main>

      {status && (
        <footer className="status-bar">
          <div className={`status ${status.startsWith('error') ? 'error' : 'info'}`}>
            {status}
          </div>
        </footer>
      )}
    </div>
  );
}

export default App;
