use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct FrameExtractionConfig {
    video_path: String,
    output_dir: String,
    interval: i32,
    prefix: String,
    start_idx: i32,
    max_idx: Option<i32>,
    format: String, // "jpg" or "png"
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("hello, {}! you've been greeted from rust!", name)
}

#[tauri::command]
async fn extract_frames(config: FrameExtractionConfig) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        extract_frames_impl(config)
    }).await.map_err(|e| format!("task error: {}", e))?
}

fn extract_frames_impl(config: FrameExtractionConfig) -> Result<String, String> {
    // Create output directory
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| format!("failed to create output directory: {}", e))?;
    
    let message = format!(
        "frame extraction configured: video={}, output={}, interval={}, prefix={}, start={}, max={:?}, format={}",
        config.video_path, config.output_dir, config.interval, config.prefix, config.start_idx, config.max_idx, config.format
    );
    
    println!("{}", message);
    Ok(format!("placeholder: would extract frames from {} to {}", config.video_path, config.output_dir))
}

#[tauri::command]
async fn split_dataset(dataset_path: String, train_ratio: f64) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        split_dataset_impl(dataset_path, train_ratio)
    }).await.map_err(|e| format!("task error: {}", e))?
}

fn split_dataset_impl(dataset_path: String, train_ratio: f64) -> Result<String, String> {
    // TODO: Implement dataset splitting
    Ok(format!("placeholder: would split dataset at {} with ratio {}", dataset_path, train_ratio))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet, extract_frames, split_dataset])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
