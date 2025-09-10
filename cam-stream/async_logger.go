package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"
)

// LogLevel defines the log levels
type LogLevel int

const (
	LevelDebug LogLevel = iota
	LevelInfo
	LevelWarn
	LevelError
)

// LogEntry represents a single log entry
type LogEntry struct {
	Level     LogLevel
	Message   string
	Timestamp time.Time
	Fields    map[string]interface{}
}

// AsyncLogger provides asynchronous logging functionality
type AsyncLogger struct {
	logChan    chan LogEntry
	done       chan struct{}
	wg         sync.WaitGroup
	logger     *slog.Logger
	bufferSize int
}

// NewAsyncLogger creates a new async logger with specified buffer size
func NewAsyncLogger(bufferSize int) *AsyncLogger {
	if bufferSize <= 0 {
		bufferSize = 1000 // default buffer size
	}

	// Create a slog logger with JSON handler
	handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	})
	
	logger := slog.New(handler)

	asyncLogger := &AsyncLogger{
		logChan:    make(chan LogEntry, bufferSize),
		done:       make(chan struct{}),
		logger:     logger,
		bufferSize: bufferSize,
	}

	// Start the background worker
	asyncLogger.wg.Add(1)
	go asyncLogger.worker()

	return asyncLogger
}

// worker processes log entries in the background
func (al *AsyncLogger) worker() {
	defer al.wg.Done()

	batch := make([]LogEntry, 0, 100) // batch size for efficiency
	ticker := time.NewTicker(100 * time.Millisecond) // flush every 100ms
	defer ticker.Stop()

	for {
		select {
		case entry, ok := <-al.logChan:
			if !ok {
				// Channel closed, flush remaining logs and exit
				al.flushBatch(batch)
				return
			}
			batch = append(batch, entry)
			
			// Flush batch if it gets too large
			if len(batch) >= 100 {
				al.flushBatch(batch)
				batch = batch[:0] // clear batch but keep capacity
			}

		case <-ticker.C:
			// Periodic flush
			if len(batch) > 0 {
				al.flushBatch(batch)
				batch = batch[:0]
			}

		case <-al.done:
			// Flush remaining logs and exit
			close(al.logChan)
			// Process remaining entries in channel
			for entry := range al.logChan {
				batch = append(batch, entry)
			}
			al.flushBatch(batch)
			return
		}
	}
}

// flushBatch writes a batch of log entries
func (al *AsyncLogger) flushBatch(entries []LogEntry) {
	for _, entry := range entries {
		al.writeLogEntry(entry)
	}
}

// writeLogEntry writes a single log entry to the underlying logger
func (al *AsyncLogger) writeLogEntry(entry LogEntry) {
	ctx := context.Background()
	
	// Convert fields to slog attributes
	attrs := make([]slog.Attr, 0, len(entry.Fields))
	for key, value := range entry.Fields {
		attrs = append(attrs, slog.Any(key, value))
	}

	switch entry.Level {
	case LevelDebug:
		al.logger.LogAttrs(ctx, slog.LevelDebug, entry.Message, attrs...)
	case LevelInfo:
		al.logger.LogAttrs(ctx, slog.LevelInfo, entry.Message, attrs...)
	case LevelWarn:
		al.logger.LogAttrs(ctx, slog.LevelWarn, entry.Message, attrs...)
	case LevelError:
		al.logger.LogAttrs(ctx, slog.LevelError, entry.Message, attrs...)
	}
}

// log is the internal method to queue a log entry
func (al *AsyncLogger) log(level LogLevel, msg string, fields map[string]interface{}) {
	entry := LogEntry{
		Level:     level,
		Message:   msg,
		Timestamp: time.Now(),
		Fields:    fields,
	}

	select {
	case al.logChan <- entry:
		// Successfully queued
	default:
		// Channel is full, drop the log entry (or could implement overflow handling)
		fmt.Fprintf(os.Stderr, "async logger buffer full, dropping log: %s\n", msg)
	}
}

// Debug logs a debug message
func (al *AsyncLogger) Debug(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	al.log(LevelDebug, msg, f)
}

// Info logs an info message
func (al *AsyncLogger) Info(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	al.log(LevelInfo, msg, f)
}

// Warn logs a warning message
func (al *AsyncLogger) Warn(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	al.log(LevelWarn, msg, f)
}

// Error logs an error message
func (al *AsyncLogger) Error(msg string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	al.log(LevelError, msg, f)
}

// Close gracefully shuts down the async logger
func (al *AsyncLogger) Close() {
	close(al.done)
	al.wg.Wait()
}

// Global async logger instance
var globalAsyncLogger *AsyncLogger
var loggerOnce sync.Once

// GetGlobalAsyncLogger returns the global async logger instance
func GetGlobalAsyncLogger() *AsyncLogger {
	loggerOnce.Do(func() {
		globalAsyncLogger = NewAsyncLogger(2000) // 2000 entry buffer
	})
	return globalAsyncLogger
}

// Package-level convenience functions that use the global async logger
func AsyncDebug(msg string, fields ...map[string]interface{}) {
	GetGlobalAsyncLogger().Debug(msg, fields...)
}

func AsyncInfo(msg string, fields ...map[string]interface{}) {
	GetGlobalAsyncLogger().Info(msg, fields...)
}

func AsyncWarn(msg string, fields ...map[string]interface{}) {
	GetGlobalAsyncLogger().Warn(msg, fields...)
}

func AsyncError(msg string, fields ...map[string]interface{}) {
	GetGlobalAsyncLogger().Error(msg, fields...)
}

// CloseGlobalAsyncLogger closes the global async logger
func CloseGlobalAsyncLogger() {
	if globalAsyncLogger != nil {
		globalAsyncLogger.Close()
	}
}
