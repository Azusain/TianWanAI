use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{command, AppHandle, Runtime};

use crate::dataset::DatasetManager;
use crate::format_converter::FormatConverter;
use crate::video_processor::VideoProcessor;

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetAnalysisRequest {
    pub dataset_path: String,
    pub images_dir: Option<String>,
    pub labels_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetSplitRequest {
    pub dataset_path: String,
    pub output_path: String,
    pub train_ratio: f64,
    pub split_mode: String,
    pub seed: u64,
    pub images_dir: Option<String>,
    pub labels_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VisualizeSamplesRequest {
    pub dataset_path: String,
    pub output_dir: Option<String>,
    pub sample_count: usize,
    pub images_dir: Option<String>,
    pub labels_dir: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VocToYoloRequest {
    pub voc_dir: String,
    pub output_dir: String,
    pub classes_file: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VideoFrameExtractionRequest {
    pub video_path: String,
    pub output_folder: String,
    pub prefix: String,
    pub target_frames: Option<u32>,
    pub interval: Option<u32>,
    pub output_format: String,
    pub quality: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchVideoExtractionRequest {
    pub input_dir: String,
    pub output_base: String,
    pub target_frames: Option<u32>,
    pub interval: Option<u32>,
    pub prefix: String,
    pub output_format: String,
    pub quality: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub message: String,
    pub data: Option<T>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T, message: &str) -> Self {
        Self {
            success: true,
            message: message.to_string(),
            data: Some(data),
        }
    }

    pub fn error(message: &str) -> Self {
        Self {
            success: false,
            message: message.to_string(),
            data: None,
        }
    }
}

// dataset management commands
#[command]
pub async fn analyze_dataset(request: DatasetAnalysisRequest) -> Result<ApiResponse<serde_json::Value>, String> {
    let manager = DatasetManager::new(&request.dataset_path)
        .map_err(|e| format!("failed to create dataset manager: {}", e))?;

    match manager.analyze_dataset(request.images_dir.as_deref(), request.labels_dir.as_deref()).await {
        Ok(analysis) => Ok(ApiResponse::success(analysis, "dataset analysis completed successfully")),
        Err(e) => Ok(ApiResponse::error(&format!("dataset analysis failed: {}", e))),
    }
}

#[command]
pub async fn split_dataset(request: DatasetSplitRequest) -> Result<ApiResponse<String>, String> {
    let manager = DatasetManager::new(&request.dataset_path)
        .map_err(|e| format!("failed to create dataset manager: {}", e))?;

    match manager.split_dataset(
        &request.output_path,
        request.train_ratio,
        &request.split_mode,
        request.seed,
        request.images_dir.as_deref(),
        request.labels_dir.as_deref(),
    ).await {
        Ok(summary) => Ok(ApiResponse::success(summary, "dataset split completed successfully")),
        Err(e) => Ok(ApiResponse::error(&format!("dataset split failed: {}", e))),
    }
}

#[command]
pub async fn visualize_samples(request: VisualizeSamplesRequest) -> Result<ApiResponse<Vec<String>>, String> {
    let manager = DatasetManager::new(&request.dataset_path)
        .map_err(|e| format!("failed to create dataset manager: {}", e))?;

    match manager.visualize_samples(
        request.output_dir.as_deref(),
        request.sample_count,
        request.images_dir.as_deref(),
        request.labels_dir.as_deref(),
    ).await {
        Ok(files) => Ok(ApiResponse::success(files, "sample visualization completed successfully")),
        Err(e) => Ok(ApiResponse::error(&format!("sample visualization failed: {}", e))),
    }
}

// format conversion commands
#[command]
pub async fn convert_voc_to_yolo(request: VocToYoloRequest) -> Result<ApiResponse<String>, String> {
    let converter = FormatConverter::new();

    match converter.convert_voc_to_yolo(&request.voc_dir, &request.output_dir, &request.classes_file).await {
        Ok(summary) => Ok(ApiResponse::success(summary, "voc to yolo conversion completed successfully")),
        Err(e) => Ok(ApiResponse::error(&format!("voc to yolo conversion failed: {}", e))),
    }
}

#[command]
pub async fn batch_convert_voc_to_yolo(voc_dirs: Vec<String>, output_base: String, classes_file: String) -> Result<ApiResponse<String>, String> {
    let converter = FormatConverter::new();

    match converter.batch_convert_voc_to_yolo(&voc_dirs, &output_base, &classes_file).await {
        Ok(summary) => Ok(ApiResponse::success(summary, "batch voc to yolo conversion completed successfully")),
        Err(e) => Ok(ApiResponse::error(&format!("batch voc to yolo conversion failed: {}", e))),
    }
}

// video processing commands
#[command]
pub async fn extract_video_frames(request: VideoFrameExtractionRequest) -> Result<ApiResponse<String>, String> {
    let processor = VideoProcessor::new(&request.output_format, request.quality)
        .map_err(|e| format!("failed to create video processor: {}", e))?;

    match processor.extract_frames(
        &request.video_path,
        &request.output_folder,
        &request.prefix,
        request.target_frames,
        request.interval,
    ).await {
        Ok(count) => Ok(ApiResponse::success(
            format!("extracted {} frames", count),
            "video frame extraction completed successfully"
        )),
        Err(e) => Ok(ApiResponse::error(&format!("video frame extraction failed: {}", e))),
    }
}

#[command]
pub async fn batch_extract_video_frames(request: BatchVideoExtractionRequest) -> Result<ApiResponse<String>, String> {
    let processor = VideoProcessor::new(&request.output_format, request.quality)
        .map_err(|e| format!("failed to create video processor: {}", e))?;

    match processor.process_directory(
        &request.input_dir,
        &request.output_base,
        request.target_frames,
        request.interval,
        &request.prefix,
    ).await {
        Ok(total_frames) => Ok(ApiResponse::success(
            format!("extracted {} total frames", total_frames),
            "batch video frame extraction completed successfully"
        )),
        Err(e) => Ok(ApiResponse::error(&format!("batch video frame extraction failed: {}", e))),
    }
}

// utility commands
#[command]
pub async fn select_directory<R: Runtime>(
    _app: AppHandle<R>,
    title: Option<String>,
) -> Result<Option<String>, String> {
    use tauri::api::dialog::blocking::FileDialogBuilder;

    let dialog = FileDialogBuilder::new()
        .set_title(&title.unwrap_or_else(|| "select directory".to_string()));

    Ok(dialog.pick_folder().map(|p| p.to_string_lossy().to_string()))
}

#[command]
pub async fn select_file<R: Runtime>(
    _app: AppHandle<R>,
    title: Option<String>,
    filters: Option<Vec<(String, Vec<String>)>>,
) -> Result<Option<String>, String> {
    use tauri::api::dialog::blocking::FileDialogBuilder;

    let mut dialog = FileDialogBuilder::new()
        .set_title(&title.unwrap_or_else(|| "select file".to_string()));

    if let Some(filters) = filters {
        for (name, extensions) in filters {
            dialog = dialog.add_filter(&name, &extensions);
        }
    }

    Ok(dialog.pick_file().map(|p| p.to_string_lossy().to_string()))
}

#[command]
pub async fn get_directory_info(path: String) -> Result<ApiResponse<serde_json::Value>, String> {
    use crate::utils::get_directory_info as get_info;

    match get_info(&path).await {
        Ok(info) => Ok(ApiResponse::success(info, "directory info retrieved successfully")),
        Err(e) => Ok(ApiResponse::error(&format!("failed to get directory info: {}", e))),
    }
}
