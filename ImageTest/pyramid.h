#pragma once
#include "image.h"

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~GuassianPyramid Class, Holder of image pyramids~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
//~~~~Image Convolution:
//https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution
//place the centre(origin) of kernel on the current pixel. 
//Then kernel will be overlapped with neighboring pixels too. 
//If kernel is symmetric (ex:Guassian kernel) then 
//multiply each kernel element with the pixel value it overlapped with and add all the obtained values. 
//Resultant value will be the value for the current pixel that is overlapped with the center of the kernel.
//If the kernel is not symmetric, it has to be flipped around its horizontal and vertical axes before doing the convolution as above

//~~~~Edge Hanlding: (see convolution link) Kernel convolution usually requires values from pixels outside of the image boundaries.
//		Extend - The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. 
//				Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.
//		Wrap - The image is conceptually wrapped (or tiled) and values are taken from the opposite edge or corner.
//		Crop - Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

//PRECOMPUTE KERNELS AND MAKE THEM STATIC MEMBERS? NO NEED TO KEEP RECALCULATING AND STORING AGAIN FOR EACH OBJECT
//JUST HAVE A PLACE IN MEM THAT ALL OBJ CAN ACCESS
//MAKE THE ABOVE STATIC MEMBERS? C++ can't do floating point static members unless it's done like below:
//class MY_CONSTS
//{
//public :
//    static const long   LONG_CONST = 1;      // Compiles 
//    static const float FLOAT_CONST;
//};
//
//const float MY_CONSTS::FLOAT_CONST = 0.001f;

//typedef struct kernel {
//	//access offset from center value i.e. mykernel(-1,2) will get 1 to the left and 2 down from center
//	__forceinline float& operator()(const int x, const int y)       { return data[dim*(centery + y) + (centerx + x)]; }
//	__forceinline float  operator()(const int x, const int y) const { return data[dim*(centery + y) + (centerx + x)]; }
//	//assumed to be square ex: 5x5 dim = 5
//	int dim,centerx,centery;
//
//	std::vector<float> data;
//} kernel;


class Pyramid {
public:
	Image pyramid[PYRAMID_LEVELS];
	//std::vector<float> kernels[PYRAMID_LEVELS - 1];
	//int kerneldims[PYRAMID_LEVELS - 1];

public:
	Pyramid();
	~Pyramid();
	Pyramid(const std::string& fileloc);
	//makes a blank pyramid
	Pyramid(const std::string& fileloc, const int w, const int h, const int bpp, const IMAGEFORMAT fmt);

	void GenPyramid(const std::string& fileloc);
	void GenLevel(const int level);
	feature ConvolveEdge(const Image& i, const int level, const int px, const int py);
	feature Convolve(const Image& i, const int level, const int px, const int py);
	void SubsampleAndSave(const Image& blurred, const int level);
	//void GenKernels();
	void PrintKernels();
	void ModifyFileNames();
	//int GetNumPyramidLevels() { return PYRAMID_LEVELS; };

	// IO
	bool Write(const FEATURETYPE& f = RGB) const;
	bool Write(const int level, const FEATURETYPE& f = RGB) const;

	//access
	__forceinline Image&        operator[](const int level)		  { return pyramid[level]; }
	__forceinline const Image&  operator[](const int level) const { return pyramid[level]; }

	__forceinline features&       operator()(const int level, const int x, const int y)       { return pyramid[level](x, y); }
	__forceinline const features& operator()(const int level, const int x, const int y) const { return pyramid[level](x, y); }

	__forceinline feature&       operator()(const int level, const FEATURETYPE f, const int x, const int y)       { return pyramid[level](f, x, y); }
	__forceinline const feature& operator()(const int level, const FEATURETYPE f, const int x, const int y) const { return pyramid[level](f, x, y); }
};

Pyramid::Pyramid() 
	: pyramid{}
{
	//GenKernels();
	//PrintKernels();
}

Pyramid::~Pyramid() {

}

Pyramid::Pyramid(const std::string& fileloc) 
	: Pyramid()
{
	GenPyramid(fileloc);
}

//For generating a blank pyramid to be filled in later
Pyramid::Pyramid(const std::string& fileloc, const int w, const int h, const int bpp, const IMAGEFORMAT fmt) 
	: Pyramid()
{
	int thewidth = w;
	int theheight = h;
	pyramid[0] = Image(fileloc, thewidth, theheight, bpp, fmt);
	for (int level = 1; level < PYRAMID_LEVELS; ++level) {
		thewidth = round(thewidth / 2.f);
		theheight = round(theheight / 2.f);
		pyramid[level] = Image(fileloc, thewidth, theheight, bpp, fmt);
	}
	ModifyFileNames();
}

void Pyramid::GenPyramid(const std::string& fileloc) {
	pyramid[0] = Image(fileloc);
	pyramid[0].GenFeature(ALL, 0);
	for (int level = 1; level < PYRAMID_LEVELS; ++level) {
		GenLevel(level);
	}
	ModifyFileNames();
}

void Pyramid::GenLevel(const int level) {
	const int parentlevel = level - 1;
	const Image& i = pyramid[parentlevel];
	Image blurred(i.filepath,i.width,i.height,i.bpp,i.format);
	const int width = i.width;
	const int height = i.height;
	//fast divide by 2
	const int edge = KERNELDIMS[parentlevel] >> 1;

	for (int py = 0; py < height; ++py) {
		for (int px = 0; px < width; ++px) {
			//TODO: maybe an enum for blur modes like TOROIDIAL vs EXTEND_EDGES
			if (px - edge < 0 || py - edge < 0 || px > i.width-1 - edge || py > i.height-1 - edge) {
				blurred(RGB, px, py) = ConvolveEdge(i, level, px, py);
			} else {
				blurred(RGB, px, py) = Convolve(i, level, px, py);
			}
		}
	}
	SubsampleAndSave(blurred, level);
	pyramid[level].GenFeature(ALL, level);
}

feature Pyramid::ConvolveEdge(const Image& i, const int level, const int px, const int py) {
	const int parentlevel = level - 1;
	//const std::vector<float>& k = kernels[parentlevel];
	//const int dim = kerneldims[parentlevel];
	const std::vector<float>& k = KERNELS[parentlevel];
	const int dim = KERNELDIMS[parentlevel];
	feature result(0, 0, 0);
	//fast divide by 2
	const int edge = dim >> 1;
	const int indexcenter = edge;


	for (int y = -edge; y <= edge; ++y) {
		for (int x = -edge; x <= edge; ++x) {
			int samplex = px + x;
			int sampley = py + y;
			int clamped = 0;
			clamp(samplex, 0, i.width - 1, clamped);
			clamp(sampley, 0, i.height - 1, clamped);

			float kernelval = k[dim*(indexcenter + y) + (indexcenter + x)];
			result += (kernelval * i(RGB, samplex, sampley));
		}
	}

	return result;
}


feature Pyramid::Convolve(const Image& i, const int level, const int px, const int py) {
	const int parentlevel = level - 1;
	//const std::vector<float>& k = kernels[parentlevel];
	//const int dim = kerneldims[parentlevel];
	const std::vector<float>& k = KERNELS[parentlevel];
	const int dim = KERNELDIMS[parentlevel];
	feature result(0, 0, 0);
	//fast divide by 2
	const int edge = dim >> 1;
	const int indexcenter = edge;

	for (int y = -edge; y <= edge; ++y) {
		for (int x = -edge; x <= edge; ++x) {
			float kernelval = k[dim*(indexcenter + y) + (indexcenter + x)];
			//Q: why didn't it recognize the compiler recognize the static version of the * override?
			//Q: why did the rhs ref get corrupted in the body of the * override when calling the [] feature operator
			result += (kernelval * i(RGB, px + x, py + y));
		}
	}

	return result;
}

void Pyramid::SubsampleAndSave(const Image& blurred, const int level) {
	const int width = blurred.width;
	const int height = blurred.height;
	this->pyramid[level] = Image(blurred.filepath, round(width/2.f), round(height/2.f), blurred.bpp, blurred.format);
	Image& i = this->pyramid[level];

	for (int y = 0; y < height; y += 2) {
		for (int x = 0; x < width; x += 2) {
			//Q: why did the assignment override in feature need rhs.data[] and not rhs[]
			i(RGB, x >> 1, y >> 1) = blurred(RGB, x, y);
		}
	}

}

//void Pyramid::GenKernels() {
//	for (int i = 0; i < PYRAMID_LEVELS - 1; ++i) {
//		//divide by a power of 2 and add 1(or round) i.e. 9->5->3
//		const int dim = round(STARTING_KERNEL_SIZE / (float)(1 << i));
//#if SCALESIGMA == 1
//		const float SIGMA = round(STARTSIGMA / (float)(1 << i));
//#else
//		const float SIGMA = STARTSIGMA;
//#endif
//		const float INV_TWOSIG2 = 1.f / (2.f * SIGMA * SIGMA);
//		const float INV_TWOPI_SIG2 = 1.f / (2.f * PI * SIGMA * SIGMA);
//		this->kerneldims[i] = dim;
//		const int size = dim*dim;
//		this->kernels[i].reserve(size);
//		this->kernels[i].resize(size);
//
//		std::vector<float>& k = this->kernels[i];
//
//		//formula requires x,y coords to be in terms of offset from center of the kernel
//		const int edge = dim >> 1;
//
//		//need this for indexing back into our array
//		const int indexcenter = edge;
//
//		//need sum of all values in kernel to normalize
//		float sum = 0.f;
//
//		for (int y = -edge; y <= edge; ++y) {
//			for (int x = -edge; x <= edge; ++x) {
//				float value = INV_TWOPI_SIG2 * pow(EULERSNUM, -1.f * (x*x + y*y) * INV_TWOSIG2);
//				k[dim*(indexcenter + y) + (indexcenter + x)] = value;
//				sum += value;
//			}
//		}
//
//		//normalize, divided by sum of all values in kernel must add to 1 so image does not brighten
//		float norm_factor = 1.f / sum;
//		for (int y = 0; y < dim; ++y) {
//			for (int x = 0; x < dim; ++x) {
//				k[dim*y + x] *= norm_factor;
//			}
//		}
//	}//for i < PYRAMID_LEVELS-1
//}

void Pyramid::PrintKernels() {
	freopen("DEBUG.txt", "w", stdout);

	for (int lvl = 0; lvl < PYRAMID_LEVELS; ++lvl) {
		//const int dim = this->kerneldims[lvl];
		const int dim = KERNELDIMS[lvl];
		std::cout << "\nKERNEL " << lvl;

		for (int y = 0; y < dim; ++y) {
			std::cout << "\n| ";
			for (int x = 0; x < dim; ++x) {
				//std::cout << " " << this->kernels[lvl][dim*y + x];
				std::cout << " " << KERNELS[lvl][dim*y + x];
			}
			std::cout << "  |";
		}

	}//lvl
}

void Pyramid::ModifyFileNames() {
	const std::string s = this->pyramid[0].filepath;

	const int dotpos = s.rfind(".");//rfind searches for last occurance, find searches for first occurance
	const int lastFslashpos = s.rfind("/",dotpos);//second arg tells it to search before that postition
	const std::string root = s.substr(0,lastFslashpos+1);//add 1 to include the slash
	const std::string filename = s.substr(lastFslashpos+1, dotpos-lastFslashpos-1);
	const std::string file_ext = s.substr(dotpos);//from dotpos to null terminator
	
	//const int dotpos = s.rfind(".");
	//const std::string root = s.substr(0,dotpos);
	//const std::string file_ext = s.substr(dotpos);
	for (int i = 0; i < PYRAMID_LEVELS; ++i) {
		this->pyramid[i].filepath = root + "lvl" + std::to_string(i) + filename + file_ext;
	}
}

bool Pyramid::Write(const FEATURETYPE& f) const {
	for (int i = 0; i < PYRAMID_LEVELS; ++i) {
		this->pyramid[i].Write(std::string(""),f);//empty string means just write to its own filepath
	}
	return true;
}

bool Pyramid::Write(const int level, const FEATURETYPE& f) const {
	this->pyramid[level].Write(std::string(""), f);
	return true;
}


