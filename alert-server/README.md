# Alert Platform Mock Server

This is a simple HTTP server that simulates the alert management platform. It receives detection alerts from the cam-stream system and saves them as JSON files for testing purposes.

## Project Structure

```
alert-server/
├── main.go              # Alert server (main application)
├── cmd/
│   └── test-client/
│       └── main.go      # Test client for sending sample alerts
├── go.mod
├── go.sum
└── README.md
```

## Features

- Receives alerts via POST /alert endpoint
- Saves each alert as a separate JSON file
- Provides web interface for monitoring
- Lists all received alerts
- Shows server statistics
- Separate test client for testing functionality

## Usage

### Start the Alert Server

```bash
# Build and run the server
go build
./alert-server.exe

# Or run directly
go run main.go
```

### Test the Server

Use the test client to send a sample alert:

```bash
# Build and run the test client
cd cmd/test-client
go build
./test-client.exe

# Or run directly
go run main.go
```

### Server Endpoints

The server will start on port 8080:
- Main interface: http://localhost:8080
- Alert endpoint: http://localhost:8080/alert
- Status: http://localhost:8080/status
- Alert list: http://localhost:8080/alerts

### Configuration in cam-stream

Configure your global alert server URL in cam-stream to:
```
http://localhost:8080/alert
```

## Alert Format

The server expects alerts in this JSON format:

```json
{
  "image": "base64_encoded_image",
  "request_id": "unique-request-id",
  "model": "model-name-v1.0",
  "camera_kks": "camera-identifier",
  "score": 0.95,
  "x1": 0.1,
  "y1": 0.2,
  "x2": 0.4,
  "y2": 0.6,
  "timestamp": "2024-01-01T12:00:00+08:00"
}
```

## Output

All received alerts are saved in the `received_alerts/` directory as JSON files with descriptive filenames containing:
- Timestamp
- Camera KKS
- Model type
- Request ID

Example filename: `20240103_143052_000_cam_1_yolo_a1b2c3d4-e5f6-7890-abcd-ef1234567890.json`
