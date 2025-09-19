//go:build linux

package rtsp

/*
#cgo pkg-config: libavformat libavcodec libavutil
#cgo CFLAGS: -std=c11

#include <stdlib.h>
#include <string.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>

typedef struct {
    int width;
    int height;
    int success;
    char error_message[256];
} DetectionResult;

DetectionResult detect_resolution(const char* url) {
    DetectionResult result = {0};
    AVFormatContext* fmt_ctx = NULL;
    int video_stream_index = -1;

    if (!url) {
        result.success = 0;
        strncpy(result.error_message, "invalid URL provided", sizeof(result.error_message) - 1);
        return result;
    }

    // initialize FFmpeg
    av_log_set_level(AV_LOG_ERROR);
    avformat_network_init();

    #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
    #endif

    // set RTSP client options
    AVDictionary* options = NULL;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "stimeout", "10000000", 0);
    av_dict_set(&options, "user_agent", "stream_detector", 0);
    av_dict_set(&options, "rtsp_flags", "prefer_tcp", 0);
    av_dict_set(&options, "buffer_size", "1000000", 0);

    // open input
    if (avformat_open_input(&fmt_ctx, url, NULL, &options) < 0) {
        av_dict_free(&options);
        result.success = 0;
        strncpy(result.error_message, "failed to open input stream", sizeof(result.error_message) - 1);
        return result;
    }

    // find stream info
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        avformat_close_input(&fmt_ctx);
        av_dict_free(&options);
        result.success = 0;
        strncpy(result.error_message, "failed to find stream info", sizeof(result.error_message) - 1);
        return result;
    }

    // find video stream
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }

    if (video_stream_index == -1) {
        avformat_close_input(&fmt_ctx);
        av_dict_free(&options);
        result.success = 0;
        strncpy(result.error_message, "no video stream found", sizeof(result.error_message) - 1);
        return result;
    }

    // get resolution
    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;
    int width = codecpar->width;
    int height = codecpar->height;

    // cleanup
    avformat_close_input(&fmt_ctx);
    av_dict_free(&options);

    if (width <= 0 || height <= 0) {
        result.success = 0;
        snprintf(result.error_message, sizeof(result.error_message), "invalid resolution: %dx%d", width, height);
        return result;
    }

    result.width = width;
    result.height = height;
    result.success = 1;
    result.error_message[0] = '\0';

    return result;
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func GetResolution(rtspUrl string) (int, int, error) {
	cURL := C.CString(rtspUrl)
	defer C.free(unsafe.Pointer(cURL))

	result := C.detect_resolution(cURL)

	if result.success == 0 {
		errMsg := C.GoString(&result.error_message[0])
		return 0, 0, fmt.Errorf("detection failed: %s", errMsg)
	}

	return int(result.width), int(result.height), nil
}
