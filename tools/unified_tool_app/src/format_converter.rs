use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use walkdir::WalkDir;

#[derive(Debug)]
pub struct VocAnnotation {
    pub name: String,
    pub bbox: [i32; 4], // [xmin, ymin, xmax, ymax]
}

#[derive(Debug)]
pub struct VocImage {
    pub width: i32,
    pub height: i32,
    pub objects: Vec<VocAnnotation>,
}

pub struct FormatConverter {
    // converter state can be stored here if needed
}

impl FormatConverter {
    pub fn new() -> Self {
        Self {}
    }

    async fn parse_voc_xml(&self, xml_path: &Path) -> Result<VocImage> {
        let content = fs::read_to_string(xml_path).await?;
        let doc = roxmltree::Document::parse(&content)?;

        // extract size information
        let size_node = doc
            .descendants()
            .find(|n| n.has_tag_name("size"))
            .ok_or_else(|| anyhow!("no size element found"))?;

        let width: i32 = size_node
            .descendants()
            .find(|n| n.has_tag_name("width"))
            .and_then(|n| n.text())
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| anyhow!("invalid width in XML"))?;

        let height: i32 = size_node
            .descendants()
            .find(|n| n.has_tag_name("height"))
            .and_then(|n| n.text())
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| anyhow!("invalid height in XML"))?;

        // extract object annotations
        let mut objects = Vec::new();
        for object_node in doc.descendants().filter(|n| n.has_tag_name("object")) {
            let name = object_node
                .descendants()
                .find(|n| n.has_tag_name("name"))
                .and_then(|n| n.text())
                .ok_or_else(|| anyhow!("object missing name"))?;

            let bndbox = object_node
                .descendants()
                .find(|n| n.has_tag_name("bndbox"))
                .ok_or_else(|| anyhow!("object missing bndbox"))?;

            let xmin: i32 = bndbox
                .descendants()
                .find(|n| n.has_tag_name("xmin"))
                .and_then(|n| n.text())
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| anyhow!("invalid xmin"))?;

            let ymin: i32 = bndbox
                .descendants()
                .find(|n| n.has_tag_name("ymin"))
                .and_then(|n| n.text())
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| anyhow!("invalid ymin"))?;

            let xmax: i32 = bndbox
                .descendants()
                .find(|n| n.has_tag_name("xmax"))
                .and_then(|n| n.text())
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| anyhow!("invalid xmax"))?;

            let ymax: i32 = bndbox
                .descendants()
                .find(|n| n.has_tag_name("ymax"))
                .and_then(|n| n.text())
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| anyhow!("invalid ymax"))?;

            objects.push(VocAnnotation {
                name: name.to_string(),
                bbox: [xmin, ymin, xmax, ymax],
            });
        }

        Ok(VocImage {
            width,
            height,
            objects,
        })
    }

    async fn load_classes(&self, classes_file: &str) -> Result<HashMap<String, i32>> {
        let content = fs::read_to_string(classes_file).await?;
        let mut classes = HashMap::new();

        for (idx, line) in content.lines().enumerate() {
            let class_name = line.trim();
            if !class_name.is_empty() {
                classes.insert(class_name.to_string(), idx as i32);
            }
        }

        Ok(classes)
    }

    fn voc_to_yolo(&self, voc_image: &VocImage, classes: &HashMap<String, i32>) -> Result<Vec<String>> {
        let mut yolo_lines = Vec::new();

        for obj in &voc_image.objects {
            let class_id = classes
                .get(&obj.name)
                .ok_or_else(|| anyhow!("unknown class: {}", obj.name))?;

            let [xmin, ymin, xmax, ymax] = obj.bbox;

            // convert to YOLO format (normalized center coordinates and dimensions)
            let center_x = (xmax + xmin) as f64 * 0.5 / voc_image.width as f64;
            let center_y = (ymax + ymin) as f64 * 0.5 / voc_image.height as f64;
            let width = (xmax - xmin) as f64 / voc_image.width as f64;
            let height = (ymax - ymin) as f64 / voc_image.height as f64;

            let line = format!("{} {:.6} {:.6} {:.6} {:.6}", class_id, center_x, center_y, width, height);
            yolo_lines.push(line);
        }

        Ok(yolo_lines)
    }

    pub async fn convert_voc_to_yolo(
        &self,
        voc_dir: &str,
        output_dir: &str,
        classes_file: &str,
    ) -> Result<String> {
        log::info!("starting voc to yolo conversion...");
        log::info!("voc directory: {}", voc_dir);
        log::info!("output directory: {}", output_dir);
        log::info!("classes file: {}", classes_file);

        // load class names
        let classes = self.load_classes(classes_file).await?;
        log::info!("loaded {} classes", classes.len());

        // create output directory
        let output_path = PathBuf::from(output_dir);
        fs::create_dir_all(&output_path).await?;

        // find all XML files in VOC directory
        let mut xml_files = Vec::new();
        for entry in WalkDir::new(voc_dir).max_depth(1) {
            let entry = entry?;
            if entry.path().extension().and_then(|ext| ext.to_str()) == Some("xml") {
                xml_files.push(entry.into_path());
            }
        }

        log::info!("found {} XML files", xml_files.len());

        let mut converted_count = 0;
        let mut error_count = 0;

        for xml_file in xml_files {
            let filename = xml_file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            match self.parse_voc_xml(&xml_file).await {
                Ok(voc_image) => {
                    match self.voc_to_yolo(&voc_image, &classes) {
                        Ok(yolo_lines) => {
                            let yolo_file = output_path.join(format!("{}.txt", filename));
                            let content = yolo_lines.join("\n");
                            if let Err(e) = fs::write(&yolo_file, content).await {
                                log::error!("failed to write {}: {}", yolo_file.display(), e);
                                error_count += 1;
                            } else {
                                converted_count += 1;
                                if converted_count <= 5 || converted_count % 10 == 0 {
                                    log::info!("converted {}: {} objects", filename, voc_image.objects.len());
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("failed to convert {}: {}", filename, e);
                            error_count += 1;
                        }
                    }
                }
                Err(e) => {
                    log::error!("failed to parse {}: {}", xml_file.display(), e);
                    error_count += 1;
                }
            }
        }

        let summary = format!(
            "conversion completed:\n- converted: {} files\n- errors: {} files\n- output: {}",
            converted_count, error_count, output_dir
        );

        log::info!("voc to yolo conversion finished!");
        log::info!("converted: {}, errors: {}", converted_count, error_count);

        Ok(summary)
    }

    pub async fn batch_convert_voc_to_yolo(
        &self,
        voc_dirs: &[String],
        output_base: &str,
        classes_file: &str,
    ) -> Result<String> {
        log::info!("starting batch voc to yolo conversion for {} directories", voc_dirs.len());

        let mut total_converted = 0;
        let mut total_errors = 0;

        for (i, voc_dir) in voc_dirs.iter().enumerate() {
            let dir_name = PathBuf::from(voc_dir)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&format!("batch_{}", i))
                .to_string();

            let output_dir = PathBuf::from(output_base).join(format!("{}_yolo", dir_name));
            let output_dir_str = output_dir.to_string_lossy().to_string();

            log::info!("processing directory {}/{}: {}", i + 1, voc_dirs.len(), voc_dir);

            match self.convert_voc_to_yolo(voc_dir, &output_dir_str, classes_file).await {
                Ok(summary) => {
                    log::info!("batch item {} completed", i + 1);
                    // extract numbers from summary if needed
                    total_converted += 1; // simplified for now
                }
                Err(e) => {
                    log::error!("batch item {} failed: {}", i + 1, e);
                    total_errors += 1;
                }
            }
        }

        let summary = format!(
            "batch conversion completed:\n- processed directories: {}\n- successful: {}\n- failed: {}\n- output base: {}",
            voc_dirs.len(),
            voc_dirs.len() - total_errors,
            total_errors,
            output_base
        );

        log::info!("batch voc to yolo conversion finished!");
        Ok(summary)
    }
}
