#include <iostream>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1; // Silent fail for Go parsing
    }
    
    const char* url = argv[1];
    AVFormatContext* fmt_ctx = nullptr;
    int video_stream_index = -1;
    
    // Initialize FFmpeg (enable error logs for debugging)
    av_log_set_level(AV_LOG_ERROR);
    
    // Initialize network for both old and new versions
    avformat_network_init();
    
    // Initialize FFmpeg for older versions only
    #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
    #endif
    
    // Set correct RTSP client options
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);     // TCP transport
    av_dict_set(&options, "stimeout", "10000000", 0);      // 10 seconds socket timeout (microseconds)
    av_dict_set(&options, "user_agent", "stream_detector", 0);
    av_dict_set(&options, "rtsp_flags", "prefer_tcp", 0);
    av_dict_set(&options, "buffer_size", "1000000", 0);
    
    // Open input
    if (avformat_open_input(&fmt_ctx, url, nullptr, &options) < 0) {
        av_dict_free(&options);
        return 1;
    }
    
    // Find stream info
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        av_dict_free(&options);
        return 1;
    }
    
    // Find video stream
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    
    if (video_stream_index == -1) {
        avformat_close_input(&fmt_ctx);
        av_dict_free(&options);
        return 1;
    }
    
    // Get resolution
    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;
    int width = codecpar->width;
    int height = codecpar->height;
    
    // Cleanup
    avformat_close_input(&fmt_ctx);
    av_dict_free(&options);
    
    if (width <= 0 || height <= 0) {
        return 1;
    }
    
    // Output: WIDTH HEIGHT (space-separated for easy Go parsing)
    std::cout << width << " " << height << std::endl;
    
    return 0;
}
