#include <iostream>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/log.h>
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <rtsp_url>" << std::endl;
        return 1;
    }
    
    const char* url = argv[1];
    AVFormatContext* fmt_ctx = nullptr;
    int video_stream_index = -1;
    
    std::cout << "Debug: Testing RTSP URL: " << url << std::endl;
    
    // Initialize FFmpeg with verbose logs for debugging
    av_log_set_level(AV_LOG_INFO);
    
    // Initialize network for both old and new versions
    avformat_network_init();
    std::cout << "Debug: Network initialized" << std::endl;
    
    // Initialize FFmpeg for older versions only
    #if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
    std::cout << "Debug: av_register_all() called (old FFmpeg)" << std::endl;
    #else
    std::cout << "Debug: Using modern FFmpeg (no av_register_all needed)" << std::endl;
    #endif
    
    // Set correct RTSP client options
    AVDictionary* options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);     // TCP transport
    av_dict_set(&options, "stimeout", "10000000", 0);      // 10 seconds socket timeout (microseconds)
    av_dict_set(&options, "user_agent", "stream_detector", 0);
    av_dict_set(&options, "rtsp_flags", "prefer_tcp", 0);
    av_dict_set(&options, "buffer_size", "1000000", 0);
    
    std::cout << "Debug: RTSP options set" << std::endl;
    
    // Open input
    std::cout << "Debug: Opening RTSP input..." << std::endl;
    int ret = avformat_open_input(&fmt_ctx, url, nullptr, &options);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "Error: Failed to open input: " << errbuf << std::endl;
        av_dict_free(&options);
        return 1;
    }
    
    std::cout << "Debug: Successfully opened input" << std::endl;
    
    // Find stream info
    std::cout << "Debug: Finding stream info..." << std::endl;
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "Error: Failed to find stream info: " << errbuf << std::endl;
        avformat_close_input(&fmt_ctx);
        av_dict_free(&options);
        return 1;
    }
    
    std::cout << "Debug: Found stream info, total streams: " << fmt_ctx->nb_streams << std::endl;
    
    // Dump format info for debugging
    av_dump_format(fmt_ctx, 0, url, 0);
    
    // Find video stream
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        AVMediaType codec_type = fmt_ctx->streams[i]->codecpar->codec_type;
        std::cout << "Debug: Stream " << i << " type: " << av_get_media_type_string(codec_type) << std::endl;
        
        if (codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            std::cout << "Debug: Found video stream at index " << i << std::endl;
            break;
        }
    }
    
    if (video_stream_index == -1) {
        std::cerr << "Error: No video stream found" << std::endl;
        avformat_close_input(&fmt_ctx);
        av_dict_free(&options);
        return 1;
    }
    
    // Get resolution
    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;
    int width = codecpar->width;
    int height = codecpar->height;
    
    std::cout << "Debug: Video codec parameters:" << std::endl;
    std::cout << "  - Width: " << width << std::endl;
    std::cout << "  - Height: " << height << std::endl;
    std::cout << "  - Codec ID: " << codecpar->codec_id << std::endl;
    std::cout << "  - Format: " << codecpar->format << std::endl;
    
    // Cleanup
    avformat_close_input(&fmt_ctx);
    av_dict_free(&options);
    avformat_network_deinit();
    
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid resolution " << width << "x" << height << std::endl;
        return 1;
    }
    
    // Output: WIDTH HEIGHT (space-separated for easy Go parsing)
    std::cout << "Result: " << width << " " << height << std::endl;
    
    return 0;
}
