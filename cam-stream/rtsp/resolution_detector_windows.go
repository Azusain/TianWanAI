//go:build windows

package rtsp

func GetResolution(rtspUrl string) (int, int, error) {
	panic("method not supported on Windows")
}
