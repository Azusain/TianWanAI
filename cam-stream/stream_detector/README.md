# Stream Detector

A simple C++ utility to detect RTSP stream resolution using FFmpeg libraries.

## Build

```bash
# In WSL/Linux
cd stream_detector
mkdir build
cd build
cmake ..
make

# The executable will be generated as: stream_detector
```

## Usage

```bash
./stream_detector rtsp://192.168.31.48:8554/test
```

Output format: `WIDTH HEIGHT` (space-separated integers)

Example output:
```
1320 1080
```

## Dependencies

- FFmpeg development libraries (libavformat, libavcodec, etc.)
- CMake 3.10+
- pkg-config

### Install dependencies (Ubuntu/WSL):

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config
sudo apt install libavformat-dev libavcodec-dev libavdevice-dev libavfilter-dev libavutil-dev libswscale-dev libswresample-dev
```

## Integration with Go

The Go program calls this utility and parses the output:

```go
cmd := exec.Command("./stream_detector/build/stream_detector", rtspURL)
output, err := cmd.Output()
if err != nil {
    return 0, 0, err
}

parts := strings.Fields(strings.TrimSpace(string(output)))
if len(parts) != 2 {
    return 0, 0, fmt.Errorf("invalid output format")
}

width, _ := strconv.Atoi(parts[0])
height, _ := strconv.Atoi(parts[1])
```
