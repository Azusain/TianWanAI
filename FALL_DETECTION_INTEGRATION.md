# TianWan Temporal Fall Detection Integration

## Overview

This document describes the integration of the Human-Falling-Detect-Tracks ST-GCN model into the TianWan API system for temporal fall detection.

## Architecture

### Components

1. **TemporalFallDetectionService** (`fall_detection_temporal.py`)
   - Maintains temporal sequences of pose keypoints per person per camera
   - Uses ST-GCN model for temporal action recognition
   - Fallback to pose-based detection when ST-GCN is unavailable
   - Manages person tracking and alert cooldown

2. **Updated API Endpoints** (`main.py`)
   - `/fall` (POST) - Process frame for fall detection
   - `/fall/status` (GET) - Get service status and statistics

3. **Submodule Integration** (`fall-detection/`)
   - Human-Falling-Detect-Tracks as Git submodule
   - Contains ST-GCN model and pose processing utilities

## Key Features

### Temporal Analysis
- Maintains 30-frame sequences per person for ST-GCN analysis
- Automatic person tracking across frames
- Temporal consistency for accurate fall detection

### Multi-Camera Support
- Independent tracking per camera
- Camera-specific statistics and alerts
- Configurable per-camera processing

### Robust Detection
- ST-GCN model for primary detection when available
- Pose-based fallback detection for reliability
- Alert cooldown to prevent spam
- Confidence thresholding

### Device Flexibility
- CUDA GPU acceleration when available
- DirectML support for Windows GPU
- CPU fallback for compatibility

## API Usage

### Fall Detection Request

```bash
POST /fall
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "camera_id": "camera_001"  // optional, defaults to "default"
}
```

### Response Format

```json
{
    "log_id": "uuid",
    "errno": 0,
    "err_msg": "SUCCESS",
    "api_version": "1.0.0",
    "model_version": "1.0.0",
    "camera_id": "camera_001",
    "timestamp": 1640995200.123,
    "persons_detected": 2,
    "fall_detected": true,
    "debug_info": {
        "model_type": "ST-GCN",
        "sequence_lengths": {
            "person_0": 30,
            "person_1": 15
        }
    },
    "results": [
        {
            "score": 0.85,
            "person_id": "person_0",
            "alert_type": "FALL_DETECTED",
            "location": {
                "left": 0.2,
                "top": 0.3,
                "width": 0.4,
                "height": 0.6
            }
        }
    ]
}
```

### Status Endpoint

```bash
GET /fall/status?camera_id=camera_001
```

```json
{
    "log_id": "uuid",
    "errno": 0,
    "err_msg": "SUCCESS",
    "api_version": "1.0.0",
    "status": {
        "version": "1.0.0",
        "model_loaded": true,
        "device": "cuda",
        "sequence_length": 30,
        "confidence_threshold": 0.7,
        "camera_stats": {
            "active_persons": 2,
            "recent_alerts": 1
        }
    }
}
```

## Integration with Cam-Stream Backend

### Frame Rate Requirements
The cam-stream backend has been updated to support 30 FPS capture:

- `config.json`: `frame_rate: 30`
- `config.go`: Default frame rate set to 30 FPS
- `rtsp_ffmpeg_cmd.go`: Uses configurable frame rate

### Processing Flow
1. Cam-stream captures frames at 30 FPS
2. Frames sent to fall detection API via HTTP POST
3. Temporal service maintains 30-frame sequences
4. ST-GCN analysis triggered when sequence is complete
5. Alerts generated for detected falls

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_fall_detection.txt
```

### 2. Initialize Submodule

```bash
git submodule update --init --recursive
```

### 3. Download Models

Place the ST-GCN model file at:
```
fall-detection/Models/TSSTG-model.pth
```

### 4. Start Service

```bash
export MODEL=fall
python main.py
```

## Configuration

### Service Parameters
- `sequence_length`: 30 frames (1 second at 30 FPS)
- `confidence_threshold`: 0.7 for fall alerts
- `alert_cooldown`: 3 seconds between alerts per person
- `max_persons`: 5 maximum tracked persons per camera

### Device Selection
The service automatically selects the best available device:
1. CUDA GPU (if available)
2. DirectML (Windows GPU)
3. CPU (fallback)

## Monitoring and Debugging

### Log Messages
- Fall detection events logged as warnings
- Model loading and device selection logged as info
- Errors logged with full stack traces

### Debug Information
Each response includes debug info:
- Model type (ST-GCN vs Pose-based)
- Sequence lengths per person
- Processing statistics

### Status Monitoring
Use `/fall/status` endpoint to monitor:
- Service health
- Active persons per camera
- Recent alert counts
- Model loading status

## Performance Considerations

### Resource Usage
- **CPU**: Moderate usage for pose extraction
- **Memory**: ~100MB per camera for sequence storage
- **GPU**: Optional but recommended for ST-GCN inference

### Optimization Tips
1. Adjust `max_persons` based on expected scene occupancy
2. Use GPU acceleration for better performance
3. Monitor sequence lengths to ensure proper tracking
4. Configure alert cooldown based on use case

## Troubleshooting

### Common Issues

1. **ST-GCN model not found**
   - Falls back to pose-based detection
   - Check model path: `fall-detection/Models/TSSTG-model.pth`

2. **Poor detection accuracy**
   - Ensure 30 FPS frame rate in cam-stream
   - Check person tracking consistency
   - Verify pose keypoint quality

3. **High memory usage**
   - Reduce `max_persons` limit
   - Check for memory leaks in long-running processes
   - Monitor sequence cleanup

4. **Import errors**
   - Initialize git submodule
   - Install required dependencies
   - Check Python path configuration

## Future Enhancements

1. **Multi-person tracking improvements**
   - Better person re-identification
   - Cross-camera tracking
   - Tracking persistence across occlusions

2. **Model optimization**
   - TensorRT acceleration
   - Model quantization
   - Batch processing

3. **Alert system**
   - Webhook notifications
   - Alert prioritization
   - Historical alert analysis

4. **Configuration management**
   - Per-camera settings
   - Dynamic parameter adjustment
   - Configuration API endpoints
