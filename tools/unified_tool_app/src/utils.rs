use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;
use tokio::fs;

pub async fn get_directory_info(path: &str) -> Result<Value> {
    let path = Path::new(path);
    
    if !path.exists() {
        return Ok(json!({
            "exists": false,
            "error": "directory does not exist"
        }));
    }
    
    if !path.is_dir() {
        return Ok(json!({
            "exists": true,
            "is_directory": false,
            "error": "path is not a directory"
        }));
    }

    let mut file_count = 0;
    let mut dir_count = 0;
    let mut total_size = 0u64;
    let mut file_types = std::collections::HashMap::new();

    let mut entries = fs::read_dir(path).await?;
    while let Some(entry) = entries.next_entry().await? {
        let entry_path = entry.path();
        
        if entry_path.is_dir() {
            dir_count += 1;
        } else if entry_path.is_file() {
            file_count += 1;
            
            if let Ok(metadata) = entry.metadata().await {
                total_size += metadata.len();
            }
            
            if let Some(extension) = entry_path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    *file_types.entry(ext_str.to_lowercase()).or_insert(0) += 1;
                }
            } else {
                *file_types.entry("no_extension".to_string()).or_insert(0) += 1;
            }
        }
    }

    Ok(json!({
        "exists": true,
        "is_directory": true,
        "path": path.display().to_string(),
        "file_count": file_count,
        "directory_count": dir_count,
        "total_size_bytes": total_size,
        "total_size_mb": total_size as f64 / (1024.0 * 1024.0),
        "file_types": file_types
    }))
}
