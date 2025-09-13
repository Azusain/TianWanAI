#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::Manager;

mod commands;
mod dataset;
mod format_converter;
mod video_processor;
mod utils;

use commands::*;

fn main() {
    env_logger::init();
    
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // dataset management commands
            analyze_dataset,
            split_dataset,
            visualize_samples,
            
            // format conversion commands
            convert_voc_to_yolo,
            batch_convert_voc_to_yolo,
            
            // video processing commands
            extract_video_frames,
            batch_extract_video_frames,
            
            // utility commands
            select_directory,
            select_file,
            get_directory_info
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_window("main").unwrap();
                window.open_devtools();
                window.close_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
