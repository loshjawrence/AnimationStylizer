#pragma once
#include "globals_utils.h"

//random number generator
#include "pcg32.h"

//popular easy to use image library by Sean Barrett aka "nothings"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


//A "feature" is a term that comes from Image Analogies. 
//It describes something about a pixel i.e. RGB, YIQ(luminance), greyscale, normal, etc.


//feature and features used directly in Image.h
//should convert all features to array of 3 floats as we may have stuff thats not 0-255 like normals, velocity, rotation?
//probably better to use a union but never used one before.
typedef struct feature {
	float data[NUM_CHANNELS];
	//access individual value at index i
    __forceinline float&       operator[](const int i)       { return data[i]; }
    __forceinline float        operator[](const int i) const { return data[i]; }

	feature() : data{0,0,0} {}
	feature(const float r, const float g, const float b) : data{ r, g, b } {}
	
	//would need to make NUM_CHANNELS a #define to #if #elif these overrides for 3 channels vs. 4 channels
	feature& operator=(const feature& rhs) {
		this->data[0] = rhs.data[0];
		this->data[1] = rhs.data[1];
		this->data[2] = rhs.data[2];
		return *this;
	}

	const feature operator+(const feature& rhs) const {
		return { this->data[0] + rhs.data[0], this->data[1] + rhs.data[1], this->data[2] + rhs.data[2] };
	}

	const feature operator-(const feature& rhs) const {
		return { this->data[0] - rhs.data[0], this->data[1] - rhs.data[1], this->data[2] - rhs.data[2] };
	}

	feature& operator+=(const feature& rhs)  {
		this->data[0] += rhs.data[0];
		this->data[1] += rhs.data[1];
		this->data[2] += rhs.data[2];
		return (*this);
	}

	feature& operator+=(const float rhs)  { //USED FOR HEATMAPS
		if (this->data[2] < 1.f && this->data[1] == 0.f && this->data[0] == 0.f) {
			this->data[2] += rhs;
		} else if (this->data[1] < 1.f && this->data[0] == 0.f ) {
			this->data[1] += rhs;
			this->data[2] -= rhs;
		} else if (this->data[0] < 1.f && this->data[2] == 0.f) {
			this->data[0] += rhs;
			this->data[1] -= rhs;
			this->data[2] = 0.f;
		} else if (this->data[0] == 1.f) {
			this->data[1] += rhs;
			this->data[2] += rhs;
		} 
		int clamped = 0;
		clamp(this->data[0], 0.f, 1.f, clamped);
		clamp(this->data[1], 0.f, 1.f, clamped);
		clamp(this->data[2], 0.f, 1.f, clamped);
		return (*this);
	}

	float SqrAndSum() const {
		return (this->data[0] * this->data[0] +
				this->data[1] * this->data[1] +
				this->data[2] * this->data[2]);
	}


} feature;

//didn't recognize the static member function version for some reason
feature operator*(const float lhs, const feature& rhs) {
	return{ lhs * rhs.data[0], lhs * rhs.data[1], lhs * rhs.data[2] };
}

typedef struct features {
	//access feature of type f
    __forceinline feature&       operator[](const FEATURETYPE f)       { return data[f]; }
    __forceinline const feature& operator[](const FEATURETYPE f) const { return data[f]; }

	features& operator=(const features& rhs) {
		for (int i = 0; i < NUM_FEATURES; ++i) {
			this->data[i] = rhs.data[i];
		}
		return *this;
	}

	features operator-(const features& rhs) {
		features f;
		for (int i = 0; i < NUM_FEATURES; ++i) {
			f.data[i] = this->data[i] - rhs.data[i];
		}
		return f;
	}

	feature data[NUM_FEATURES];
} features;

features operator*(const float lhs, const features& rhs) {
	features f;
	for (int i = 0; i < NUM_FEATURES; ++i) {
		f.data[i] = lhs * rhs.data[i];
	}
	return f;
}


/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~Image Class, Holder of Pixels~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
class Image {
public:
	int width, height, bpp;
	std::vector<features> pixelfeatures;
	std::string filepath;
	IMAGEFORMAT format;
	float meanLuminance; // used for luminance remapping
	float stdDevLuminance; 

public:
	//constructors
	Image();
	~Image();
	Image(const std::string& fileloc);
	Image(const std::string& fileloc, const int w, const int h, const int bytesperpixel, const IMAGEFORMAT fmt);
	Image(const Image& i);

	//feature(s) manip
	bool GenFeature(const FEATURETYPE f, const int level);
	bool GenFeatureALL_EXCEPTFLOW(const int level);
	bool GenFeatureYIQ();
	bool GenFeatureDX2DY2DXDY();
	bool GenFeatureORI(const int level);
	bool GenFeatureFLOW(const Image* const otherframe, const int level, const FEATURETYPE f);
	void CalculateStandardDeviation();
	bool RemapLuminance(const float sigmaA, const float sigmaB, const float muA, const float muB);

	//IO
	bool Read();
	bool Write(std::string& fileloc = std::string(""), const FEATURETYPE& f = RGB) const ;

	//utility
	void determineFormat(std::string& fileloc, bool read);
	void changeFileName(std::string& filename);
	void colorByLocation();
	void CopyImageRGBToOpenCVMatAndConvertToGray(cv::Mat& opencvmat) const;
	void CopyImageYToOpenCVMatSingleChannel8bit(cv::Mat& opencvmat) const;
	void CopyOpenCVMatInfoToImageFeatures(const cv::Mat& opencvmat, const FEATURETYPE f);
	void CopyOpenCVMatInfoToORIFeatures(const cv::Mat& opencvmat);
	void CopyOpenCVMatInfoToFLOWFeatures(const cv::Mat& opencvmat, const FEATURETYPE f);
	//Operator overrides
	//access features for the pixel at pixel index(w,h)
	__forceinline       features& operator()(const int x, const int y)       { return pixelfeatures[y*width + x]; }
	__forceinline const features& operator()(const int x, const int y) const { return pixelfeatures[y*width + x]; }
	//access individual feature of type f at pixel index(w,h)
	__forceinline       feature& operator()(const FEATURETYPE f, const int x, const int y)       { return pixelfeatures[y*width + x][f]; }
	__forceinline const feature& operator()(const FEATURETYPE f, const int x, const int y) const { return pixelfeatures[y*width + x][f]; }
	//return type is reference for cascade assignment, needed?
	Image& operator=(const Image& rhs);
};

Image::Image()
	: width(0), height(0), bpp(0), pixelfeatures(0), filepath(), format(JPG), meanLuminance(0), stdDevLuminance(0)
{

}

Image::~Image() {

}

Image::Image(const std::string& fileloc) 
	: width(0), height(0), bpp(0), pixelfeatures(0), filepath(fileloc), format(JPG), meanLuminance(0), stdDevLuminance(0)
{
	Read();
}
Image::Image(const std::string& fileloc, const int w, const int h, const int bytesperpixel, const IMAGEFORMAT fmt)
	: width(w), height(h), bpp(bytesperpixel), pixelfeatures(w*h), filepath(fileloc), format(fmt), meanLuminance(0), stdDevLuminance(0)
{

}

Image::Image(const Image& i) 
	: width(i.width), height(i.height), bpp(i.bpp), pixelfeatures(i.pixelfeatures), filepath(i.filepath), format(i.format), meanLuminance(i.meanLuminance), stdDevLuminance(i.stdDevLuminance)
{

}

Image& Image::operator=(const Image& rhs) {
	if (this == &rhs) {
		return *this;
	} else {
		width = rhs.width;
		height = rhs.height;
		bpp = rhs.bpp;
		pixelfeatures = rhs.pixelfeatures;
		filepath = rhs.filepath;
		meanLuminance = rhs.meanLuminance;
		stdDevLuminance = rhs.stdDevLuminance;
		return *this;
	}
}

void Image::determineFormat(std::string& fileloc, bool read) {
	//determine file extension
	if (read) {
		format = PNG;
	} else {
		format = PNG;
	}
}

void Image::changeFileName(std::string& filename) {

	const int dotpos = filepath.rfind(".");//rfind searches for last occurance, find searches for first occurance
	const int lastFslashpos = filepath.rfind("/",dotpos);//second arg tells it to search before that postition
	const std::string root = filepath.substr(0,lastFslashpos+1);//add 1 to include the slash
	const std::string file_ext = filepath.substr(dotpos);//from dotpos to null terminator
	filepath = root + filename + file_ext;
}

void Image::colorByLocation() {
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const float xpercent = ((float)x) / (width - 1);
			const float ypercent = ((float)y) / (height - 1);
			feature color(xpercent, 0, ypercent);
			if ( y == 0 ) {
				color = feature(1,1,1);
			} else if ( y == (height - 1) ) {
				color = feature(0,1,1);
			} else if ( x == 0 ) {
				color = feature(1,1,0);
			} else if ( x == (width - 1) ) {
				color = feature(0,1,0);
			}
			(*this)(RGB, x, y) = color;
		}
	}
}

bool Image::Read() {
	determineFormat(filepath,true);
	uint8* rgb_image = stbi_load(filepath.c_str(), &width, &height, &bpp, NUM_CHANNELS);

	if (rgb_image == NULL) {
		printf("\nstbi_load error: %s\n", stbi_failure_reason());
	}

	//makes the vector the right size, avoids constantly re-allocating
	//the internal array if you were to use push_back 
	pixelfeatures.reserve(width*height);
	pixelfeatures.resize(width*height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int rindex = y*width*NUM_CHANNELS + x*NUM_CHANNELS;
			uint8 r = rgb_image[rindex];
			uint8 g = rgb_image[rindex + 1];
			uint8 b = rgb_image[rindex + 2];
			feature rgb(r / 255.f, g / 255.f, b / 255.f);
			(*this)(RGB, x, y) = rgb;
		}
	}

    stbi_image_free(rgb_image);
	return true;
}

bool Image::Write(std::string& fileloc, const FEATURETYPE& f) const {
	if ( fileloc == std::string("") ) {//wasn't allowed to set the default to this->filepath in the funciton dec :(
		fileloc = this->filepath;
	}
	//determineFormat(fileloc,false);
	uint8* rgb_image = (uint8*)malloc(width * height * NUM_CHANNELS);

	if (f == RGB) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int rindex = NUM_CHANNELS * (y*width + x);
				const feature& rgb = (*this)(RGB, x, y);
				rgb_image[rindex] = (uint8)(rgb[0] * 255);
				rgb_image[rindex + 1] = (uint8)(rgb[1] * 255);
				rgb_image[rindex + 2] = (uint8)(rgb[2] * 255);
			}
		}
	}
	else if (f == YIQ) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int rindex = NUM_CHANNELS * (y*width + x);
				const float Y = (*this)(YIQ, x, y)[0];
				rgb_image[rindex]     = (uint8)(Y * 255);
				rgb_image[rindex + 1] = (uint8)(Y * 255);
				rgb_image[rindex + 2] = (uint8)(Y * 255);
			}
		}
	}
	else if (f == DX2DY2DXY) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int rindex = NUM_CHANNELS * (y*width + x);
				feature rgb = (*this)(DX2DY2DXY, x, y);
				if (rgb[0] < 0) rgb[0] *= -1.f;
				if (rgb[1] < 0) rgb[1] *= -1.f;
				if (rgb[2] < 0) rgb[2] *= -1.f;
				int clamped = 0;
				clamp(rgb[0], 0, 1, clamped);
				clamp(rgb[1], 0, 1, clamped);
				clamp(rgb[2], 0, 1, clamped);

				rgb_image[rindex] = (uint8)(rgb[0] * 255);
				rgb_image[rindex + 1] = (uint8)(rgb[1] * 255);
				rgb_image[rindex + 2] = (uint8)(0 * 255);
			}
		}
	}
	else if (f == ORI) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int rindex = NUM_CHANNELS * (y*width + x);
				feature rgb = (*this)(ORI, x, y);
				int clamped = 0;
				clamp(rgb[0], -1, 1, clamped);
				clamp(rgb[1], -1, 1, clamped);
				rgb[0] = (rgb[0] + 1.f) / 2.f;
				rgb[1] = (rgb[1] + 1.f) / 2.f;
				rgb_image[rindex] = (uint8)(rgb[0] * 255);
				rgb_image[rindex + 1] = (uint8)(rgb[1] * 255);
				rgb_image[rindex + 2] = (uint8)(0 * 255);
			}
		}
	}
	else if (f == FLOW_FWD) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int rindex = NUM_CHANNELS * (y*width + x);
				feature rgb = (*this)(FLOW_FWD, x, y);
				int clamped = 0;
				const float range = 5;
				clamp(rgb[0], -range, range, clamped);
				clamp(rgb[1], -range, range, clamped);
				rgb[0] = (rgb[0] + range) / (2.f * range);
				rgb[1] = (rgb[1] + range) / (2.f * range);

				rgb_image[rindex]     = (uint8)(rgb[0] * 255);
				rgb_image[rindex + 1] = (uint8)(rgb[1] * 255);
				rgb_image[rindex + 2] = (uint8)(0 * 255);
			}
		}
	}
	else if (f == FLOW_BWD) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int rindex = NUM_CHANNELS * (y*width + x);
				feature rgb = (*this)(FLOW_BWD, x, y);
				int clamped = 0;
				const float range = 5;
				clamp(rgb[0], -range, range, clamped);
				clamp(rgb[1], -range, range, clamped);
				rgb[0] = (rgb[0] + range) / (2.f * range);
				rgb[1] = (rgb[1] + range) / (2.f * range);

				rgb_image[rindex]     = (uint8)(rgb[0] * 255);
				rgb_image[rindex + 1] = (uint8)(rgb[1] * 255);
				rgb_image[rindex + 2] = (uint8)(0 * 255);
			}
		}
	}

	//TODO: PNG write is a bit slow, use BMP for now
	switch (format) {
	case JPG: //JPG not supported on write, use PNG instead
		//stbi_write_png(fileloc.c_str(), width, height, NUM_CHANNELS, rgb_image, width * NUM_CHANNELS);
		stbi_write_bmp(fileloc.c_str(), width, height, NUM_CHANNELS, rgb_image);
		break;
	case PNG:
		//stbi_write_png(fileloc.c_str(), width, height, NUM_CHANNELS, rgb_image, width * NUM_CHANNELS);
		stbi_write_bmp(fileloc.c_str(), width, height, NUM_CHANNELS, rgb_image);
		break;
	case BMP:
		stbi_write_bmp(fileloc.c_str(), width, height, NUM_CHANNELS, rgb_image);
		break;
	default:
		printf("\nImage::Write() -> Unrecognized IMAGEFORMAT\n");
	}

	stbi_image_free(rgb_image);
	return true;
}

bool Image::GenFeature(const FEATURETYPE f, const int level) {
	switch (f) {
	case RGB:
		std::cout << "\nRGB should already be generated\n" + f;
		return false;
		break;
	case YIQ:
		return GenFeatureYIQ();
		break;
	case DX2DY2DXY:
		return GenFeatureDX2DY2DXDY();
		break;
	case ORI:
		return GenFeatureORI(level);
		break;
	case ALL:
		return GenFeatureALL_EXCEPTFLOW(level);
		break;

	default:
		std::cout << "\nUnrecognized FEATURETYPE\n" + f;
		return false;
	}
}

bool Image::GenFeatureALL_EXCEPTFLOW(const int level) {
	GenFeatureYIQ();
	GenFeatureDX2DY2DXDY();
	GenFeatureORI(level);
	return true;
}

bool Image::GenFeatureDX2DY2DXDY() {
	//generate dx*dx, dy*dy, and dx*dy terms for each pixel, needed for neighborhood orientation calculation
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float dx = 0;
			float dy = 0;


			for (int sy = -1; sy <= 1; ++sy) {
				for (int sx = -1; sx <= 1; ++sx) {
					const int sobelindex = (sy + 1) * 3 + (sx + 1);
					int px = x + sx;
					int py = y + sy;
					int clamped = 0;
					clamp(px, 0, width - 1, clamped);
					clamp(py, 0, height - 1, clamped);

					const float grayscalePixel = (*this)(YIQ, px, py)[0];
					dx += (grayscalePixel * SOBELKERNELFLIPPED_DX[sobelindex]);
					dy += (grayscalePixel * SOBELKERNELFLIPPED_DY[sobelindex]);
				}//sx
			}//sy
			(*this)(DX2DY2DXY, x, y) = feature(dx*dx, dy*dy, dx*dy);
		}//x
	}//y
	return true;
}

void Image::CopyImageYToOpenCVMatSingleChannel8bit(cv::Mat& opencvmat) const {
	int clamped = 0;
	for (int y = 0; y < opencvmat.rows; ++y) {
		for (int x = 0; x < opencvmat.cols; ++x) {
			float r = (*this)(YIQ, x, y)[0];
			opencvmat.at<uchar>(y, x) = (uint8_t)(r * 255.f);
		}
	}
}

void Image::CopyImageRGBToOpenCVMatAndConvertToGray(cv::Mat& opencvmat) const {
	for (int y = 0; y < opencvmat.rows; ++y) {
		for (int x = 0; x < opencvmat.cols; ++x) {
			opencvmat.at<cv::Vec3f>(y, x)[0] = (*this)(RGB, x, y)[0];
			opencvmat.at<cv::Vec3f>(y, x)[1] = (*this)(RGB, x, y)[1];
			opencvmat.at<cv::Vec3f>(y, x)[2] = (*this)(RGB, x, y)[2];
		}
	}
	cvtColor(opencvmat, opencvmat, CV_BGR2GRAY);
	//DEBUG:
    //namedWindow( "Display window next", WINDOW_AUTOSIZE ); // Create a window for display.
    //imshow( "Display window next", thiscv ); // Show our image inside it.
    //waitKey(0); // Wait for a keystroke in the window
}

		
void Image::CopyOpenCVMatInfoToORIFeatures(const cv::Mat& opencvmat) {
	for (int y = 0; y < opencvmat.rows; ++y) {
		for (int x = 0; x < opencvmat.cols; ++x) {
			const float eigenval1 = opencvmat.at<cv::Vec6f>(y, x)[0];
			const float eigenval2 = opencvmat.at<cv::Vec6f>(y, x)[1];
			//if (eigenval1 >= eigenval2) {
				(*this)(ORI, x, y)[0] = opencvmat.at<cv::Vec6f>(y, x)[2];
				(*this)(ORI, x, y)[1] = opencvmat.at<cv::Vec6f>(y, x)[3];
			//} else {
			//	(*this)(ORI, x, y)[0] = opencvmat.at<cv::Vec6f>(y, x)[4];
			//	(*this)(ORI, x, y)[1] = opencvmat.at<cv::Vec6f>(y, x)[5];
			//}
		}
	}
}

void Image::CopyOpenCVMatInfoToFLOWFeatures(const cv::Mat& opencvmat, const FEATURETYPE f) {
	for (int y = 0; y < opencvmat.rows; ++y) {
		for (int x = 0; x < opencvmat.cols; ++x) {
				(*this)(f, x, y)[0] = opencvmat.at<cv::Vec2f>(y, x)[0];
				(*this)(f, x, y)[1] = opencvmat.at<cv::Vec2f>(y, x)[1];
		}
	}
}

void Image::CopyOpenCVMatInfoToImageFeatures(const cv::Mat& opencvmat ,const FEATURETYPE f) {
	switch (f) {
	case ORI:
		CopyOpenCVMatInfoToORIFeatures(opencvmat);
		break;
	case FLOW_FWD:
		CopyOpenCVMatInfoToFLOWFeatures(opencvmat, f);
		break;
	case FLOW_BWD:
		CopyOpenCVMatInfoToFLOWFeatures(opencvmat, f);
		break;
	default:
		cout << "\nFEATURETYPE not recognized.";
	}
}


bool Image::GenFeatureFLOW(const Image* const otherframe, const int level, const FEATURETYPE f) {
	Mat flow;
	Mat thismat(height, width, CV_8UC1);
	Mat othermat(height, width, CV_8UC1);
	CopyImageYToOpenCVMatSingleChannel8bit(thismat);
	if (otherframe == nullptr) {//end frames will set otherfame to nullptr appropriately
		othermat = thismat;
	} else {
		otherframe->CopyImageYToOpenCVMatSingleChannel8bit(othermat);
	}
	const int windowdim = 31 >> level;//61?
 //calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    calcOpticalFlowFarneback(thismat, othermat, flow, 0.5, 3, windowdim, 5, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
    //calcOpticalFlowFarneback(thismat, othermat, flow, 0.5, 3, 15, 3, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
//	cvtColor(thismat, thismat, CV_GRAY2BGR);
//	cvtColor(othermat, othermat, CV_GRAY2BGR);
//	//calcOpticalFlowSF(thismat, othermat, flow, 3, 2, 4);
//	//calcOpticalFlowSF(thismat, othermat, flow, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);

	CopyOpenCVMatInfoToImageFeatures(flow, f);
	////DEBUG:
	//Mat cflow;
	//cvtColor(thismat, cflow, CV_GRAY2BGR);
	//drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));
 //   namedWindow( "OpticalFlowFarneback", WINDOW_AUTOSIZE ); // Create a window for display.
	//imshow("OpticalFlowFarneback", cflow);
 //   waitKey(0); // Wait for a keystroke in the window

	return true;
}

//bool Image::GenFeatureORI(const int level) {
//	//convert to opencv Mat, call eigen function extract data and save into image
//	const int blockSize = KERNELDIMS[level];//structure tensor neighborhood dim
//	const int ksize = 3;//sobel kernel size?
//	Mat src(height, width, CV_32FC3);
//	Mat dst;
//	CopyImageRGBToOpenCVMatAndConvertToGray(src);
//	cornerEigenValsAndVecs(src, dst, blockSize, ksize, BORDER_REPLICATE);
//	CopyOpenCVMatInfoToImageFeatures(dst, ORI);
//	return true;
//}

bool Image::GenFeatureORI(const int level) {
	const int dim = KERNELDIMS[level];
	const std::vector<float>& k = KERNELS[level];
#if STRUCTURE_TENSOR_NO_NEIGHBORHOOD == 1
	const int edge = 0;
#else
	const int edge = dim >> 1;
#endif

	//sum up dx2 dy2 dxdy in the neighborhood and weight each by the appropriate gaussian kernel value
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float sumDX2 = 0;
			float sumDY2 = 0;
			float sumDXDY = 0;

			for (int ny = -edge; ny <= edge; ++ny) {
				for (int nx = -edge; nx <= edge; ++nx) {
					const int kernelindex = (ny + edge) * dim + (nx + edge);
					int px = x + nx;
					int py = y + ny;
					int clamped = 0;
					clamp(px, 0, width - 1, clamped);
					clamp(py, 0, height - 1, clamped);

#if STRUCTURE_TENSOR_NO_NEIGHBORHOOD == 1
					const float kernelval = 1;
#else
					const float kernelval = k[kernelindex];
#endif
					sumDX2  += (kernelval * (*this)(DX2DY2DXY, px, py)[0]);
					sumDY2  += (kernelval * (*this)(DX2DY2DXY, px, py)[1]);
					sumDXDY += (kernelval * (*this)(DX2DY2DXY, px, py)[2]);
				}//nx
			}//ny

			//using the 3 summations calcualte the dominant eigen value (magnitude of the dominant gradient direction)
			//lec6.pdf in "useful links/structure tensor" folder slide 21
			//using the 3 summations and the lambdas calculate dominant gradient eigen vector
			const float h11_minus_h22 = sumDX2 - sumDY2;
			const float EIGENVALUEPLUS  = 0.5f * ((sumDX2 + sumDY2) + sqrtf(4.f * sumDXDY * sumDXDY + h11_minus_h22 * h11_minus_h22) );
			const float h11_minus_eigenplus = sumDX2 - EIGENVALUEPLUS;
			float EIGENVECTORPLUS_x = 1;
			float EIGENVECTORPLUS_y = -h11_minus_eigenplus/ sumDXDY;

//			if (x == 265 && y == 116) {//circle image with discontinuities
//				std::cout << "\ntop left side: "  << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x ==284  && y == 115) {
//				std::cout << "\ntop right side: " << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x == 112 && y == 275) {
//				std::cout << "\nleft top side: "  << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x == 112 && y == 304) {
//				std::cout << "\nleft bot side: "  << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x == 427 && y == 243) {
//				std::cout << "\nright top side: " << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x == 427 && y == 275) {
//				std::cout << "\nright bot side: " << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x == 255 && y == 434) {
//				std::cout << "\nbot left side: "  << h11_minus_eigenplus << ", " << sumDXDY;
//			} else if (x == 273 && y == 434) {
//				std::cout << "\nbot right side: " << h11_minus_eigenplus << ", " << sumDXDY << "\n";
//			} 


			if (fabs(h11_minus_eigenplus) < 0.005) { EIGENVECTORPLUS_y = 0; }
			int clamped = 0;
			clamp(EIGENVECTORPLUS_y, -10.f, 10.f, clamped);
			//normalize
			const float mag = sqrtf(EIGENVECTORPLUS_x * EIGENVECTORPLUS_x + EIGENVECTORPLUS_y * EIGENVECTORPLUS_y);
			EIGENVECTORPLUS_x /= mag;
			//EIGENVECTORPLUS_y /= mag;
			EIGENVECTORPLUS_y = fabs(EIGENVECTORPLUS_y / mag);

			//TODO: save 4 segment circle and print eigen vector values on either side of segments 
			//try the other equation for solving for eigen vector


			//comment to make flat regions have [1,0] as x basis
			if (sumDX2 < 0.005 && sumDY2 < 0.005) { //set x axis basis vector to 0 so no rotation is done between another neighborhood and this one
				EIGENVECTORPLUS_x = 0;
				EIGENVECTORPLUS_y = 0;
			}
			(*this)(ORI, x, y) = feature(EIGENVECTORPLUS_x, EIGENVECTORPLUS_y, EIGENVALUEPLUS);

//			if (x == 265 && y == 116) {//circle image with discontinuities
//				std::cout << "\ntop left side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x ==284  && y == 115) {
//				std::cout << "\ntop right side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x == 112 && y == 275) {
//				std::cout << "\nleft top side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x == 112 && y == 304) {
//				std::cout << "\nleft bot side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x == 427 && y == 243) {
//				std::cout << "\nright top side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x == 427 && y == 275) {
//				std::cout << "\nright bot side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x == 255 && y == 434) {
//				std::cout << "\nbot left side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y;
//			} else if (x == 273 && y == 434) {
//				std::cout << "\nbot right side: " << EIGENVECTORPLUS_x << ", " << EIGENVECTORPLUS_y << "\n";
//			} 
		}//x
	}//y
	return true;
}

bool Image::GenFeatureYIQ()
{
	// generate YIQ by converting RGB and store as a feature
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const feature& rgb = (*this)(RGB, x, y);
			float Y = 0.299900 * rgb[0] + 0.587000 * rgb[1] + 0.114000 * rgb[2];
			float I = 0.595716 * rgb[0] - 0.274453 * rgb[1] - 0.321264 * rgb[2];
			float Q = 0.211456 * rgb[0] - 0.522591 * rgb[1] + 0.311350 * rgb[2];
			int clamped = 0;
			clamp(Y, 0.f, 1.f, clamped);
			clamp(I, -0.5957, 0.5957, clamped); 
			clamp(Q, -0.5226, 0.5226, clamped);
			meanLuminance += Y;
			feature yiq(Y, I, Q);
			(*this)(YIQ, x, y) = yiq;
		}
	}
	meanLuminance /= (height * width);
	CalculateStandardDeviation();
	return true;
}

bool Image::RemapLuminance(const float sigmaA, const float sigmaB, const float muA, const float muB) {
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const feature& rgb = (*this)(RGB, x, y);
			//recalc a fresh Y from the rgb data b/c we are remapping with each frame (can't assume all frames of B to have the same histogram)
			//and Y will be the old remapped luminance from a previous frame if we reuse it
			const float Y = 0.2999 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
			float remappedY = ( (sigmaB / sigmaA) * (Y - muA) )+ muB;
			int clamped = 0;
			clamp(remappedY, 0.f, 1.f, clamped);
			(*this)(YIQ, x, y)[0] = remappedY;
		}
	}
	return true;
}

void Image::CalculateStandardDeviation() {
	float variance = 0;
	float difference = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			difference = ((*this)(YIQ, x, y)[0] - meanLuminance);
			variance += (difference * difference);
		}
	}
	variance /= (height*width);
	stdDevLuminance = sqrt(variance);
}

