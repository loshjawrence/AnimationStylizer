#include "stylizer.h"

int main() {
	//std::string readimage("gray/INPUT_VIDEO/FRAMES/SteamboatWillie8.bmp");
	//std::string writeimage("gray/INPUT_VIDEO/FRAMES/imagewrite.bmp");
	//Image myimage(readimage);
	//myimage.Write(writeimage);
	//Pyramid mypyramid(readimage);
	//mypyramid.Write();
          
	//std::string A("basictests/A1.png");
	//std::string Ap("basictests/Ap1.png");
	//std::string B("basictests/B1.png");
	//std::string A("gray/circle.png");
	//std::string Ap("gray/checker.png");
	//std::string B("gray/valve1.png");
	//std::string A("gray/A2.png");
	//std::string Ap("gray/Ap2.png");
	//std::string A("gray/vertstripes1.png");
	//std::string Ap("gray/vertstripes1.png");
	//std::string A("gray/tex2.png");
	//std::string Ap("gray/tex2.png");
	//std::string A("gray/bricks2.png");
	//std::string Ap("gray/bricks2.png");
	//std::string A("color/A2.png");
	//std::string Ap("color/Ap2.png");
	//std::string B("color/Bboat3.png");
	//std::string B("gray/Bface3.png");
	//std::string B("gray/Bfly3.png");
	//std::string B("gray/MAYA_ANIM/animationballcube192.bmp");
	//std::string B("gray/horizstripes1.png");
	//std::string A("color/A3starryOIL.png");
	//std::string Ap("color/Ap3starryOIL.png");
	//Stylizer(A, Ap, B);
	//std::string A("gray/A2.png");
	//std::string Ap("gray/Ap2.png");

//	std::string A("color/A3starryOIL.png");
//	std::string Ap("color/Ap3starryOIL.png");
	std::string A("color/grad2.jpg");
	std::string Ap("color/grad2.jpg");
	//disocclusion shots frame 26, go 20 frames. or try frame 100, go 40 frames
	//std::string B("gray/MAYA_ANIM/animationballcube26.bmp");
	//std::string B("C:/temp/anvilframes/asequence2/resize/output640_1173.bmp");
	std::string B("C:/temp/NoSurprisesFrames/resize/output640_603.bmp");
//	std::string B("color/face.jpg");
	//std::string B("C:/temp/NoSurprisesFrames/output435.bmp");
	Stylizer(A, Ap, B, "", 2);//0 to process all frames starting from the one selected

	//starryoil k = 3 or 4, N = 9, S = 6
	//basket k = 1, N = 9, S = 6
	//bricks k = 1, N = 9, S = 6
	//hatch k = 0.5, N = 5, S = 1

    return 0;
	//LONG one starting from 761, 500
	//LONG one starting from 812, 592
    //ffmpeg -i a%d.bmp -i b%d.bmp -filter_complex hstack output.mp4
	//https://ffmpeg.org/ffmpeg-filters.html#hstack
}

//int main( int argc, char** argv )
//{
//    //if( argc != 2)
//    //{
//    // cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
//    // return -1;
//    //}
//
//    Mat imageprev, imagenext, flow;
//	std::string B1("gray/MAYA_ANIM/animationballcube27.bmp");
//	std::string B2("gray/MAYA_ANIM/animationballcube28.bmp");
//    imageprev = imread(B1, IMREAD_GRAYSCALE); // Read the file
//    imagenext = imread(B2, IMREAD_GRAYSCALE); // Read the file
//
//    if( imageprev.empty() || imagenext.empty()) { cout << "Could not open or find the image" << std::endl; return -1; }
//
//    calcOpticalFlowFarneback(imageprev, imagenext, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//
//	//VIEW THE FIELD
//	Mat cflow;
//	cvtColor(imageprev, cflow, CV_GRAY2BGR);
//	drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));
//    namedWindow( "OpticalFlowFarneback", WINDOW_AUTOSIZE ); // Create a window for display.
//	imshow("OpticalFlowFarneback", cflow);
//
//	//VIEW THE FRAMES
//    namedWindow( "Display window prev", WINDOW_AUTOSIZE ); // Create a window for display.
//    namedWindow( "Display window next", WINDOW_AUTOSIZE ); // Create a window for display.
//    imshow( "Display window prev", imageprev ); // Show our image inside it.
//    imshow( "Display window next", imagenext ); // Show our image inside it.
//
//    waitKey(0); // Wait for a keystroke in the window
//    return 0;
//}














//int main() {
//	const int width = 100;
//	const int height = 100;
//	av_register_all(); // Loads the whole database of available codecs and formats.
//
//	struct SwsContext* convertCtx = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL); // Preparing to convert my generated RGB images to YUV frames.
//
//	// Preparing the data concerning the format and codec in order to write properly the header, frame data and end of file.
//	char *fmtext = "mp4";
//	char *filename;
//	sprintf(filename, "GeneratedVideo.%s", fmtext);
//	AVOutputFormat * fmt = av_guess_format(fmtext, NULL, NULL);
//	AVFormatContext *oc = NULL;
//	avformat_alloc_output_context2(&oc, NULL, NULL, filename);
//	AVStream * stream = avformat_new_stream(oc, 0);
//	AVCodec *codec = NULL;
//	AVCodecContext *c = NULL;
//	int ret;
//
//	codec = avcodec_find_encoder_by_name("libx264");
//
//	// Setting up the codec:
//	av_dict_set(&opt, "preset", "slow", 0);
//	av_dict_set(&opt, "crf", "20", 0);
//	avcodec_get_context_defaults3(stream->codec, codec);
//	c = avcodec_alloc_context3(codec);
//	c->width = width;
//	c->height = height;
//	c->pix_fmt = AV_PIX_FMT_YUV420P;
//
//	// Setting up the format, its stream(s), linking with the codec(s) and write the header:
//	if (oc->oformat->flags & AVFMT_GLOBALHEADER) // Some formats require a global header.
//		c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
//	avcodec_open2(c, codec, &opt);
//	av_dict_free(&opt);
//	stream->time_base = (AVRational){ 1, 25 };
//	stream->codec = c; // Once the codec is set up, we need to let the container know which codec are the streams using, in this case the only (video) stream.
//	av_dump_format(oc, 0, filename, 1);
//	avio_open(&oc->pb, filename, AVIO_FLAG_WRITE);
//	ret = avformat_write_header(oc, &opt);
//	av_dict_free(&opt);
//
//	// Preparing the containers of the frame data:
//	AVFrame *rgbpic, *yuvpic;
//
//	// Allocating memory for each RGB frame, which will be lately converted to YUV:
//	rgbpic = av_frame_alloc();
//	rgbpic->format = AV_PIX_FMT_RGB24;
//	rgbpic->width = width;
//	rgbpic->height = height;
//	ret = av_frame_get_buffer(rgbpic, 1);
//
//	// Allocating memory for each conversion output YUV frame:
//	yuvpic = av_frame_alloc();
//	yuvpic->format = AV_PIX_FMT_YUV420P;
//	yuvpic->width = width;
//	yuvpic->height = height;
//	ret = av_frame_get_buffer(yuvpic, 1);
//
//	// After the format, code and general frame data is set, we write the video in the frame generation loop:
//	// std::vector<uint8_t> B(width*height*3);
//
//
//
//	Matrix B(width, height);
//	int got_output;
//	AVPacket pkt;
//	for (i = 0; i < N; i++)
//	{
//		generateframe(B, i); // This one is the function that generates a different frame for each i.
//		// The AVFrame data will be stored as RGBRGBRGB... row-wise, from left to right and from top to bottom, hence we have to proceed as follows:
//		for (y = 0; y < height; y++)
//		{
//			for (x = 0; x < width; x++)
//			{
//				// rgbpic->linesize[0] is equal to width.
//				rgbpic->data[0][y*rgbpic->linesize[0] + 3 * x] = B(x, y)->Red;
//				rgbpic->data[0][y*rgbpic->linesize[0] + 3 * x + 1] = B(x, y)->Green;
//				rgbpic->data[0][y*rgbpic->linesize[0] + 3 * x + 2] = B(x, y)->Blue;
//			}
//		}
//		sws_scale(convertCtx, rgbpic->data, rgbpic->linesize, 0, height, yuvpic->data, yuvpic->linesize); // Not actually scaling anything, but just converting the RGB data to YUV and store it in yuvpic.
//		av_init_packet(&pkt);
//		pkt.data = NULL;
//		pkt.size = 0;
//		yuvpic->pts = i; // The PTS of the frame are just in a reference unit, unrelated to the format we are using. We set them, for instance, as the corresponding frame number.
//		ret = avcodec_encode_video2(c, &pkt, yuvpic, &got_output);
//		if (got_output)
//		{
//			fflush(stdout);
//			av_packet_rescale_ts(&pkt, (AVRational) { 1, 25 }, stream->time_base); // We set the packet PTS and DTS taking in the account our FPS (second argument) and the time base that our selected format uses (third argument).
//			pkt.stream_index = stream->index;
//			printf("Write frame %6d (size=%6d)\n", i, pkt.size);
//			av_interleaved_write_frame(oc, &pkt); // Write the encoded frame to the mp4 file.
//			av_packet_unref(&pkt);
//		}
//	}
//	// Writing the delayed frames:
//	for (got_output = 1; got_output; i++) {
//		ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
//		if (got_output) {
//			fflush(stdout);
//			av_packet_rescale_ts(&pkt, (AVRational) { 1, 25 }, stream->time_base);
//			pkt.stream_index = stream->index;
//			printf("Write frame %6d (size=%6d)\n", i, pkt.size);
//			av_interleaved_write_frame(oc, &pkt);
//			av_packet_unref(&pkt);
//		}
//	}
//	av_write_trailer(oc); // Writing the end of the file.
//	if (!(fmt->flags & AVFMT_NOFILE))
//		avio_closep(oc->pb); // Closing the file.
//	avcodec_close(stream->codec);
//	// Freeing all the allocated memory:
//	sws_freeContext(convertCtx);
//	av_frame_free(&rgbpic);
//	av_frame_free(&yuvpic);
//	avformat_free_context(oc);
//}