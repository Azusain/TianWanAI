package main

import (
	"fmt"
	"sync"
)

// FFmpegProxyManager manages multiple FFmpeg stream proxies
type FFmpegProxyManager struct {
	proxies map[string]*FFmpegStreamProxy
	config  *FFmpegProxyConfig
	mutex   sync.RWMutex
}

// NewFFmpegProxyManager creates a new proxy manager
func NewFFmpegProxyManager(config *FFmpegProxyConfig) *FFmpegProxyManager {
	if config == nil {
		config = DefaultFFmpegProxyConfig()
	}
	
	return &FFmpegProxyManager{
		proxies: make(map[string]*FFmpegStreamProxy),
		config:  config,
	}
}

// StartProxy starts a proxy for the given camera ID and RTSP URL (resolution auto-detected)
func (fpm *FFmpegProxyManager) StartProxy(cameraID, rtspURL string) (*FFmpegStreamProxy, error) {
	fpm.mutex.Lock()
	defer fpm.mutex.Unlock()
	
	// Check if proxy already exists
	if proxy, exists := fpm.proxies[cameraID]; exists {
		if proxy.IsRunning() {
			return proxy, nil
		}
		// Clean up old proxy
		proxy.Stop()
		delete(fpm.proxies, cameraID)
	}
	
	// Create and start new proxy (resolution auto-detected via ffprobe)
	proxy := NewFFmpegStreamProxy(rtspURL)
	if err := proxy.Start(); err != nil {
		return nil, fmt.Errorf("failed to start proxy for camera %s: %v", cameraID, err)
	}
	
	fpm.proxies[cameraID] = proxy
	return proxy, nil
}

// GetProxy returns the proxy for a given camera ID
func (fpm *FFmpegProxyManager) GetProxy(cameraID string) (*FFmpegStreamProxy, bool) {
	fpm.mutex.RLock()
	defer fpm.mutex.RUnlock()
	
	proxy, exists := fpm.proxies[cameraID]
	return proxy, exists
}

// StopProxy stops and removes the proxy for a given camera ID
func (fpm *FFmpegProxyManager) StopProxy(cameraID string) error {
	fpm.mutex.Lock()
	defer fpm.mutex.Unlock()
	
	proxy, exists := fpm.proxies[cameraID]
	if !exists {
		return nil
	}
	
	err := proxy.Stop()
	delete(fpm.proxies, cameraID)
	return err
}

// StopAll stops all proxies
func (fpm *FFmpegProxyManager) StopAll() error {
	fpm.mutex.Lock()
	defer fpm.mutex.Unlock()
	
	var lastErr error
	for cameraID, proxy := range fpm.proxies {
		if err := proxy.Stop(); err != nil {
			lastErr = err
		}
		delete(fpm.proxies, cameraID)
	}
	
	return lastErr
}

// GetAllProxies returns all active proxies
func (fpm *FFmpegProxyManager) GetAllProxies() map[string]*FFmpegStreamProxy {
	fpm.mutex.RLock()
	defer fpm.mutex.RUnlock()
	
	result := make(map[string]*FFmpegStreamProxy)
	for cameraID, proxy := range fpm.proxies {
		result[cameraID] = proxy
	}
	
	return result
}

// GetProxyCount returns the number of active proxies
func (fpm *FFmpegProxyManager) GetProxyCount() int {
	fpm.mutex.RLock()
	defer fpm.mutex.RUnlock()
	
	return len(fpm.proxies)
}
