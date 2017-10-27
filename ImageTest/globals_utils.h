#pragma once
#define _CRT_SECURE_NO_WARNINGS 1 
#include <vector>
#include <array>
#include <string>
#include <stdint.h>
#include <iostream>
#include <cstdio>
//defining NDEBUG will disable asserts
//#define NDEBUG
#include <assert.h>
#include "stdafx.h"
#include "ffmpegDecode.h"
#include <windows.h>
#include <direct.h>
#include <fstream>
#include <cstddef>        // std::size_t

//#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp> //farneback
#include <opencv2/imgcodecs.hpp>//reading images?
#include <opencv2/highgui.hpp>//needed for gui display of results
//#include <opencv2/optflow.hpp>
using namespace cv;
using namespace std;

//#define STB_IMAGE_RESIZE_IMPLEMENTATION
//#include "stb_image_resize.h"
//still bugs, but as it stands(bug is probably related to coherence matching image edge cases(constantly borrowing from same?):
//NPR:								USEAPRIMECOLOR 0,OVERWRITESMAP 0,SEPARATE_SMAP 1,COH_MATCH_SEARCH_COMPLETED_NEIGHBORHOOD 1,REMAPLUM 1,YMODE 1,KCONST 1 
//texture synth:					USEAPRIMECOLOR 1,OVERWRITESMAP 0,SEPARATE_SMAP 1,COH_MATCH_SEARCH_COMPLETED_NEIGHBORHOOD 1,REMAPLUM 1,YMODE 1,KCONST 1
//black background in animation:	USEAPRIMECOLOR 1,OVERWRITESMAP 1,SEPARATE_SMAP 0,COH_MATCH_SEARCH_COMPLETED_NEIGHBORHOOD 0,REMAPLUM 0,YMODE 1,KCONST 0 
#define USEAPRIMECOLOR 1 //use Ap color in output image
#define SCALEAPRIMEBACK 0 //if result is too dim or bright turn this on to scale the histo back
#define STRUCTURE_TENSOR_NO_NEIGHBORHOOD 0 
#define ROTATE_NEIGHBORHOODS 0
#define OCCLUSION_EPSILON 2.f // as defined in pixar paper, for detecting occulsion

//DEBUG
#define WRITE_MATCH_TYPE 0 //color image based on where the match came from R: CohMatch G: PatchMatch B: advected result from prev frame 
#define WRITE_MATCH_LOC  0 //color image based on location in Aprime Red is x axis Blue is Y axis
const float HEATMAP_SCALE = 1000.f;//magic number for heat map
//dark colors are close to upper left
//red is close to upper right
//blue is close to bottom left
//purple is close to bottom right

const int TEMPORALLY_COHERENT = 0;
const float KCONST = 2; //2-25 for NPR filters, 1 for line art, 0.5-5 for texture synthesis
const float ADVECTEDWEIGHT = 1.f / 2.f;
const int PYRAMID_LEVELS = 3;
#define PYRLVLS 3
const int STARTING_KERNEL_SIZE = 9;
const float PI = 3.1415926536f;
const float EULERSNUM = 2.718281828456f;
const float STARTSIGMA = 4;//use a sensible sigma for the window size(STARTING_KERNEL_SIZE)
const int PATCHMATCH_TOTAL_ITERATIONS = 2;//better results even?
const int PATCHMATCH_RANDOM_INITIALIZATIONS = 3;
typedef uint8_t uint8;


//Gaussian Kernels
int GenKernels(std::vector<float>* const kernels, int* const kerneldims) {
	//std::vector<float> kernels[PYRAMID_LEVELS - 1];
	for (int i = 0; i < PYRAMID_LEVELS - 1; ++i) {
		//divide by a power of 2 and add 1(or round) i.e. 9->5->3
		const int dim = round(STARTING_KERNEL_SIZE / (float)(1 << i));
		const float SIGMA = round(STARTSIGMA / (float)(1 << i));
		const float INV_TWOSIG2 = 1.f / (2.f * SIGMA * SIGMA);
		const float INV_TWOPI_SIG2 = 1.f / (2.f * PI * SIGMA * SIGMA);
		kerneldims[i] = dim;
		const int size = dim*dim;
		kernels[i].reserve(size);
		kernels[i].resize(size);

		std::vector<float>& k = kernels[i];

		//formula requires x,y coords to be in terms of offset from center of the kernel
		const int edge = dim >> 1;

		//need this for indexing back into our array
		const int indexcenter = edge;

		//need sum of all values in kernel to normalize
		float sum = 0.f;

		for (int y = -edge; y <= edge; ++y) {
			for (int x = -edge; x <= edge; ++x) {
				float value = INV_TWOPI_SIG2 * pow(EULERSNUM, -1.f * (x*x + y*y) * INV_TWOSIG2);
				k[dim*(indexcenter + y) + (indexcenter + x)] = value;
				sum += value;
			}
		}

		//normalize, divided by sum of all values in kernel must add to 1 so image does not brighten
		float norm_factor = 1.f / sum;
		for (int y = 0; y < dim; ++y) {
			for (int x = 0; x < dim; ++x) {
				k[dim*y + x] *= norm_factor;
			}
		}
	}//for i < PYRAMID_LEVELS-1
	return 1;
}
std::vector<float> kernelsnonconst[PYRAMID_LEVELS - 1];
int kerneldimsnonconst[PYRAMID_LEVELS - 1];
int DONOTUSE = GenKernels(&(kernelsnonconst[0]), &(kerneldimsnonconst[0]));
//no way to assign to const with out hardcoded inits, if 
#if PYRLVLS == 4
const std::vector<float> KERNELS[PYRAMID_LEVELS + 1] = { kernelsnonconst[0], kernelsnonconst[1], kernelsnonconst[2], kernelsnonconst[0] , std::vector<float>(0)};
const int KERNELDIMS[PYRAMID_LEVELS + 1] = { kerneldimsnonconst[0], kerneldimsnonconst[1], kerneldimsnonconst[2], kerneldimsnonconst[0], 0};
#elif PYRLVLS == 3
const std::vector<float> KERNELS[PYRAMID_LEVELS + 1] = { kernelsnonconst[0], kernelsnonconst[1], kernelsnonconst[0], std::vector<float>(0)};
const int KERNELDIMS[PYRAMID_LEVELS + 1] = { kerneldimsnonconst[0], kerneldimsnonconst[1], kerneldimsnonconst[0], 0};
#elif PYRLVLS == 2
const std::vector<float> KERNELS[PYRAMID_LEVELS + 1] = { kernelsnonconst[0], kernelsnonconst[0], std::vector<float>(0) };
const int KERNELDIMS[PYRAMID_LEVELS + 1] = { kerneldimsnonconst[0], kerneldimsnonconst[0], 0};
#else
#endif




//YMODE = 1 means we are going to convert RGB to YIQ and do all our image neighborhood comparisons with just the Y luminance component
//for grey scale images comparisons in RGB produce a better result, for color images Y produces a better result. (at the moment)

//SMM is the second-moment matrix aka 'structure tensor' matrix it holds partial derivative info for the neighborhood of a given pixel in an image.
//The neighborhood size for a level will be the same size as the guassian kernel for that level.
//We will use the two sobel kernels mentioned in the wiki to find the x slopes and y slopes, dx and dy. Kernels are flipped about x and y to make convolution easier
//i.e. after flipping, simply place the kernel over the pixel then multiply the kernel elements with the pixel elements that line up then sum to find the value at that pixel.
//https://en.wikipedia.org/wiki/Sobel_operator
//the flipped sobel kernel for finding dx of a pixel:
/*
[ -1 0 1 ]
[ -2 0 2 ]
[ -1 0 1 ]
*/
//the flipped sobel kernel for finding dy of a pixel:
/*
[ -1 -2 -1 ]
[  0  0  0 ]
[  1  2  1 ]
*/
//https://en.wikipedia.org/wiki/Structure_tensor
//over a the neighborhood of the pixel the structure tensor for that pixel is:
/*
[ sum(dx*dx)  sum(dx*dy) ]
[ sum(dx*dy)  sum(dy*dy) ]
*/
//in the summation, they also mention multiplying by a window weight that sums to 1 (sounds like guassian kernel to me)
//since the right diagonal is the same, we only need 3 values to store this, might as well add it to the feature vector.
//NORMALIZED KERNELS
//const float SOBELKERNELFLIPPED_DX[9] = { -1 * 0.125, 0       , 1 * 0.125,
//                                         -2 * 0.125, 0       , 2 * 0.125, 
//	                                     -1 * 0.125, 0       , 1 * 0.125 };
//
//const float SOBELKERNELFLIPPED_DY[9] = { -1 * 0.125, -2 * 0.125, -1 * 0.125, 
//                                          0,          0,          0, 
//	                                      1 * 0.125,  2 * 0.125,  1 * 0.125};
const float SOBELKERNELFLIPPED_DX[9] = { -1, 0, 1,
                                         -2, 0, 2, 
	                                     -1, 0, 1 };

const float SOBELKERNELFLIPPED_DY[9] = { -1, -2, -1, 
                                          0,  0,  0, 
	                                      1 , 2, 1};

const int NUM_FEATURES = 6;
//RGB = red green blue
//YIQ = luminance Y and color differences I and Q
//DX2DY2DXY are the dx*dx, dy*dy, and dx*dy taken from the sobel gradient images, just for that pixel, these are saved precalculations for the ORI calculation(which used the structure tensor, which is formed by a sum of these terms over the neighborhood)
//ORI is the orientation of the neighborhood for that pixel (neighborhood size is size of a Gaussian kernel at that level), normalized 2D vector pointing to its predominant gradient direction in the neighborhood
//FLOW_FWD is the forward displacement from this frame to the next for the pixel, (0,0) for the last frame
//FLOW_BWD is the backward displacemtn form this frame to the prev for the pixel, (0,0) for the first frame
enum FEATURETYPE {RGB, YIQ, DX2DY2DXY, ORI, FLOW_FWD, FLOW_BWD, ALL};

const int NUM_CHANNELS = 3; // 3 = RGB or 4 = RGBA 
enum IMAGEFORMAT {JPG,PNG,BMP};
std::array<std::string, 4> SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".flv"};//im sure theres more ffmpeg supports
std::array<std::string, 3> SUPPORTED_IMAGE_FORMATS = { ".jpg", ".png", ".bmp" };


//should use template function but w/e
void clamp(int& sample, const int min, const int max, int& clamped) {
	if (sample < min) {
		sample = min;
		clamped = 1;
	} else if (sample > max) {
		sample = max;
		clamped = 1;
	}
}
void clamp(float& sample, const float min, const float max, int& clamped) {
	if (sample < min) {
		sample = min;
		clamped = 1;
	} else if (sample > max) {
		sample = max;
		clamped = 1;
	}
}

inline bool file_exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
	} else {
		return false;
	}
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
	for (int y = 0; y < cflowmap.rows; y += step) {
		for (int x = 0; x < cflowmap.cols; x += step) {
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)), color);
			//circle(cflowmap, Point(x, y), 1, color, -1);
		}
	}
}
//// ffmpegDecoder.cpp : Defines the entry point for the console application.
////comes from github
//
////#define FILE_NAME          "C:\\temp\\lorn_segasunset.mp4"
////define OUTPUT_FILE_PREFIX "c:\\temp\\image%d.bmp"
////#define FRAME_COUNT        50
//
//bool BMPSave(const char *pFileName, AVFrame * frame, int w, int h)
//{
//	bool bResult = false;
//
//	if (frame)
//	{
//		FILE* file = fopen(pFileName, "wb");
//
//		if (file)
//		{
//			// RGB image
//			int imageSizeInBytes = 3 * w * h;
//			int headersSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
//			int fileSize = headersSize + imageSizeInBytes;
//
//			uint8_t * pData = new uint8_t[headersSize];
//
//			if (pData != NULL)
//			{
//				BITMAPFILEHEADER& bfHeader = *((BITMAPFILEHEADER *)(pData));
//
//				bfHeader.bfType = 'MB';
//				bfHeader.bfSize = fileSize;
//				bfHeader.bfOffBits = headersSize;
//				bfHeader.bfReserved1 = bfHeader.bfReserved2 = 0;
//
//				BITMAPINFOHEADER& bmiHeader = *((BITMAPINFOHEADER *)(pData + headersSize - sizeof(BITMAPINFOHEADER)));
//
//				bmiHeader.biBitCount = 3 * 8;
//				bmiHeader.biWidth    = w;
//				bmiHeader.biHeight   = h;
//				bmiHeader.biPlanes   = 1;
//				bmiHeader.biSize     = sizeof(bmiHeader);
//				bmiHeader.biCompression = BI_RGB;
//				bmiHeader.biClrImportant = bmiHeader.biClrUsed = 
//					bmiHeader.biSizeImage = bmiHeader.biXPelsPerMeter = 
//					bmiHeader.biYPelsPerMeter = 0;
//
//				fwrite(pData, headersSize, 1, file);
//
//				uint8_t *pBits = frame->data[0] + frame->linesize[0] * h - frame->linesize[0];
//				int nSpan      = -frame->linesize[0];
//
//				int numberOfBytesToWrite = 3 * w;
//
//				for (size_t i = 0; i < h; ++i, pBits += nSpan)	
//				{
//					fwrite(pBits, numberOfBytesToWrite, 1, file);
//				}
//
//				bResult = true;
//				delete [] pData;				
//			}
//
//			fclose(file);
//		}
//	}
//
//	return bResult;
//}
//
//
//void SplitVideoIntoFrames_ffmpegSOMEFRAMES(const std::string& FILE_NAME, const int FRAME_COUNT, const std::string& outputfileprefix) {
//	FFmpegDecoder decoder;
//	const char* OUTPUT_FILE_PREFIX = outputfileprefix.c_str();
//
//	if (decoder.OpenFile(std::string(FILE_NAME))) {
//		int w = decoder.GetWidth();
//		int h = decoder.GetHeight();
//
//		for (int i = 0; i < FRAME_COUNT; i++) {
//			AVFrame * frame = decoder.GetNextFrame();
//			if (frame) {
//				char filename[MAX_PATH];
//				sprintf(filename, OUTPUT_FILE_PREFIX, i);
//				if (!BMPSave(filename, frame, frame->width, frame->height)) {
//					printf("Cannot save file %s\n", filename);
//				}
//				av_free(frame->data[0]);
//				av_free(frame);
//			}
//		}
//
//		decoder.CloseFile();
//	} else {
//		printf("Cannot open file %s \n", FILE_NAME.c_str());
//	}
//}
//
//int SplitVideoIntoFrames_ffmpegALLFRAMES(const std::string& FILE_NAME, const std::string& outputfileprefix) {
//	FFmpegDecoder decoder;
//	const char* OUTPUT_FILE_PREFIX = outputfileprefix.c_str();
//
//	if (decoder.OpenFile(std::string(FILE_NAME))) {
//		int w = decoder.GetWidth();
//		int h = decoder.GetHeight();
//		AVFrame * frame = decoder.GetNextFrame();
//		int i = 0;
//		while(frame) {
//			char filename[MAX_PATH];
//			sprintf(filename, OUTPUT_FILE_PREFIX, i);
//			if (!BMPSave(filename, frame, frame->width, frame->height)) {
//				printf("Cannot save file %s\n", filename);
//			}
//			av_free(frame->data[0]);
//			av_free(frame);
//			frame = decoder.GetNextFrame();
//			i++;
//		}
//		decoder.CloseFile();
//		return i;
//	} else {
//		printf("Cannot open file %s \n", FILE_NAME.c_str());
//		return 0;
//	}
//}
