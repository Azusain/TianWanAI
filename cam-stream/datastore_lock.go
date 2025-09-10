package main

import "sync"

// 为 dataStore 添加全局锁
var dataStoreLock sync.RWMutex

// getCameraConfig 线程安全地获取摄像头配置
func getCameraConfig(cameraID string) *CameraConfig {
	dataStoreLock.RLock()
	defer dataStoreLock.RUnlock()
	
	return dataStore.Cameras[cameraID]
}
