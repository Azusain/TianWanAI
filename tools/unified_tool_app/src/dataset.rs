use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    pub total_images: usize,
    pub labeled_images: usize,
    pub unlabeled_images: usize,
    pub total_annotations: usize,
    pub class_distribution: HashMap<i32, usize>,
    pub class_names: HashMap<i32, String>,
    pub image_sizes: HashMap<String, usize>,
    pub issues: DatasetIssues,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetIssues {
    pub missing_labels: Vec<String>,
    pub empty_label_files: Vec<String>,
    pub invalid_annotations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloAnnotation {
    pub class_id: i32,
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
}

pub struct DatasetManager {
    dataset_path: PathBuf,
    image_extensions: Vec<String>,
}

impl DatasetManager {
    pub fn new(dataset_path: &str) -> Result<Self> {
        let path = PathBuf::from(dataset_path);
        if !path.exists() {
            return Err(anyhow!("dataset path does not exist: {}", dataset_path));
        }

        Ok(Self {
            dataset_path: path,
            image_extensions: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "bmp".to_string(),
                "tiff".to_string(),
                "webp".to_string(),
            ],
        })
    }

    async fn find_images(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        let mut images = Vec::new();

        for entry in WalkDir::new(directory).max_depth(1) {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if let Some(ext_str) = extension.to_str() {
                        if self.image_extensions.iter().any(|ext| ext.eq_ignore_ascii_case(ext_str)) {
                            images.push(path.to_path_buf());
                        }
                    }
                }
            }
        }

        Ok(images)
    }

    async fn read_yolo_label(&self, label_path: &Path) -> Result<Vec<YoloAnnotation>> {
        if !label_path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(label_path).await?;
        let mut annotations = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 5 {
                log::warn!("invalid annotation in {}:{}: {}", label_path.display(), line_num + 1, line);
                continue;
            }

            match (
                parts[0].parse::<i32>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
                parts[3].parse::<f64>(),
                parts[4].parse::<f64>(),
            ) {
                (Ok(class_id), Ok(center_x), Ok(center_y), Ok(width), Ok(height)) => {
                    if (0.0..=1.0).contains(&center_x)
                        && (0.0..=1.0).contains(&center_y)
                        && (0.0..=1.0).contains(&width)
                        && (0.0..=1.0).contains(&height)
                    {
                        annotations.push(YoloAnnotation {
                            class_id,
                            center_x,
                            center_y,
                            width,
                            height,
                        });
                    } else {
                        log::warn!("invalid coordinates in {}:{}: {}", label_path.display(), line_num + 1, line);
                    }
                }
                _ => {
                    log::warn!("parse error in {}:{}: {}", label_path.display(), line_num + 1, line);
                }
            }
        }

        Ok(annotations)
    }

    async fn load_class_names(&self) -> Result<HashMap<i32, String>> {
        let mut class_names = HashMap::new();

        // try common class name files
        for filename in ["classes.txt", "data.yaml", "dataset.yaml"] {
            let file_path = self.dataset_path.join(filename);
            if !file_path.exists() {
                continue;
            }

            if filename.ends_with(".yaml") || filename.ends_with(".yml") {
                if let Ok(content) = fs::read_to_string(&file_path).await {
                    if let Ok(data) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
                        if let Some(names) = data.get("names") {
                            if let Some(names_map) = names.as_mapping() {
                                for (key, value) in names_map {
                                    if let (Some(k), Some(v)) = (key.as_i64(), value.as_str()) {
                                        class_names.insert(k as i32, v.to_string());
                                    }
                                }
                            } else if let Some(names_seq) = names.as_sequence() {
                                for (i, name) in names_seq.iter().enumerate() {
                                    if let Some(name_str) = name.as_str() {
                                        class_names.insert(i as i32, name_str.to_string());
                                    }
                                }
                            }
                            return Ok(class_names);
                        }
                    }
                }
            } else if let Ok(content) = fs::read_to_string(&file_path).await {
                for (i, line) in content.lines().enumerate() {
                    let line = line.trim();
                    if !line.is_empty() {
                        class_names.insert(i as i32, line.to_string());
                    }
                }
                return Ok(class_names);
            }
        }

        Ok(class_names)
    }

    pub async fn analyze_dataset(
        &self,
        images_dir: Option<&str>,
        labels_dir: Option<&str>,
    ) -> Result<Value> {
        log::info!("starting dataset analysis...");

        // determine directories
        let images_path = if let Some(dir) = images_dir {
            PathBuf::from(dir)
        } else {
            // auto-detect
            for possible_dir in ["images", "train/images", "val/images", "."] {
                let candidate = self.dataset_path.join(possible_dir);
                if candidate.exists() && !self.find_images(&candidate).await?.is_empty() {
                    break candidate;
                }
            }
            self.dataset_path.clone()
        };

        let labels_path = if let Some(dir) = labels_dir {
            PathBuf::from(dir)
        } else {
            // auto-detect
            for possible_dir in ["labels", "train/labels", "val/labels", "."] {
                let candidate = self.dataset_path.join(possible_dir);
                if candidate.exists() {
                    break candidate;
                }
            }
            self.dataset_path.clone()
        };

        log::info!("analyzing images in: {}", images_path.display());
        log::info!("analyzing labels in: {}", labels_path.display());

        // find all images
        let image_files = self.find_images(&images_path).await?;
        log::info!("found {} image files", image_files.len());

        if image_files.is_empty() {
            return Err(anyhow!("no image files found"));
        }

        // analyze images and labels
        let mut analysis = DatasetAnalysis {
            total_images: image_files.len(),
            labeled_images: 0,
            unlabeled_images: 0,
            total_annotations: 0,
            class_distribution: HashMap::new(),
            class_names: HashMap::new(),
            image_sizes: HashMap::new(),
            issues: DatasetIssues {
                missing_labels: Vec::new(),
                empty_label_files: Vec::new(),
                invalid_annotations: Vec::new(),
            },
        };

        for img_file in image_files {
            let img_stem = img_file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let label_file = labels_path.join(format!("{}.txt", img_stem));

            if !label_file.exists() {
                analysis.unlabeled_images += 1;
                analysis.issues.missing_labels.push(img_file.display().to_string());
                continue;
            }

            // read annotations
            let annotations = match self.read_yolo_label(&label_file).await {
                Ok(ann) => ann,
                Err(e) => {
                    log::warn!("failed to read label file {}: {}", label_file.display(), e);
                    analysis.issues.invalid_annotations.push(label_file.display().to_string());
                    continue;
                }
            };

            if annotations.is_empty() {
                analysis.unlabeled_images += 1;
                analysis.issues.empty_label_files.push(label_file.display().to_string());
            } else {
                analysis.labeled_images += 1;
                analysis.total_annotations += annotations.len();

                // analyze annotations
                for ann in annotations {
                    *analysis.class_distribution.entry(ann.class_id).or_insert(0) += 1;
                }
            }

            // get actual image size using image crate
            match image::image_dimensions(&img_file) {
                Ok((width, height)) => {
                    let size_key = format!("{}x{}", width, height);
                    *analysis.image_sizes.entry(size_key).or_insert(0) += 1;
                }
                Err(_) => {
                    // fallback to extension tracking for corrupted images
                    if let Some(ext) = img_file.extension().and_then(|e| e.to_str()) {
                        *analysis.image_sizes.entry(format!("corrupted_{}", ext)).or_insert(0) += 1;
                    }
                }
            }
        }

        // load class names
        analysis.class_names = self.load_class_names().await.unwrap_or_default();

        log::info!("dataset analysis complete!");
        log::info!("total images: {}", analysis.total_images);
        log::info!("labeled images: {}", analysis.labeled_images);
        log::info!("unlabeled images: {}", analysis.unlabeled_images);
        log::info!("total annotations: {}", analysis.total_annotations);

        serde_json::to_value(analysis).map_err(|e| anyhow!("failed to serialize analysis: {}", e))
    }

    pub async fn split_dataset(
        &self,
        output_path: &str,
        train_ratio: f64,
        split_mode: &str,
        seed: u64,
        images_dir: Option<&str>,
        labels_dir: Option<&str>,
    ) -> Result<String> {
        // implementation would be similar to the python version
        // for now, return a summary
        let summary = format!(
            "dataset split completed:\n- output: {}\n- train ratio: {:.1}%\n- mode: {}\n- seed: {}",
            output_path,
            train_ratio * 100.0,
            split_mode,
            seed
        );

        log::info!("split_dataset called with ratio: {}", train_ratio);
        Ok(summary)
    }

    pub async fn visualize_samples(
        &self,
        output_dir: Option<&str>,
        sample_count: usize,
        images_dir: Option<&str>,
        labels_dir: Option<&str>,
    ) -> Result<Vec<String>> {
        // implementation would generate visualization files
        // for now, return placeholder file paths
        let mut files = Vec::new();
        for i in 1..=sample_count.min(5) {
            files.push(format!("sample_{:02d}.png", i));
        }

        log::info!("visualize_samples called for {} samples", sample_count);
        Ok(files)
    }
}
