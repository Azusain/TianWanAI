//go:build windows

package main

func GetResolution(rtspUrl string) (int, int, error) {
	panic("method not supported on Windows")
}
