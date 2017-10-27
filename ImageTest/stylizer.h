#pragma once
#include "pyramid.h"

//3rd element can be used for advection bool (0 means nothing advected here from other frame, 1 means something did)
typedef struct ivec4 {
	int data[4];
	__forceinline int&       operator[](const int i) { return data[i]; }
	__forceinline int        operator[](const int i) const { return data[i]; }

	ivec4(): data{0,0,0,0} {}

	ivec4(const int x, const int y)
		: data{x,y,0,0}
	{
	}
	ivec4(const int x, const int y, const int z)
		: data{x,y,z,0}
	{
	}
	ivec4(const int x, const int y, const int z, const int w)
		: data{x,y,z,w}
	{
	}
	
	const ivec4& operator=(const ivec4& rhs) {
		this->data[0] = rhs.data[0];
		this->data[1] = rhs.data[1];
		this->data[2] = rhs.data[2];
		this->data[3] = rhs.data[3];
		return *this;
	}

	const ivec4 operator+(const ivec4& rhs) const {
		return { this->data[0] + rhs.data[0], this->data[1] + rhs.data[1], this->data[2], this->data[3]};
	}

	const ivec4 operator-(const ivec4& rhs) const {
		return { this->data[0] - rhs.data[0], this->data[1] - rhs.data[1], this->data[2], this->data[3]};
	}

	ivec4& operator+=(const ivec4& rhs)  {
		this->data[0] += rhs.data[0];
		this->data[1] += rhs.data[1];
		return (*this);
	}

	float length() {
		return sqrtf(this->data[0] * this->data[0] + this->data[1] * this->data[1]);
	}
} ivec4;

const ivec4 operator*(const float lhs, const ivec4& rhs) {
	return{ int(rhs.data[0] * lhs), int(rhs.data[1] * lhs) };
}

//source map ('s' in Image Analogies, 'Mt' in Pixar paper)
//returns the pixel in Ap that was copied over to Bp pixel x,y 
typedef struct SourceMap {
	std::vector<ivec4> data;
	int width, height;
	const SourceMap& operator=(const SourceMap& rhs) {
		const int size = rhs.data.size();
		data.reserve(size);
		data.resize(size);
		width = rhs.width;
		height = rhs.height;
		for (int i = 0; i < size; ++i) {
			this->data[i] = rhs.data[i];
		}
		return (*this);
	}
	__forceinline ivec4& operator()(const int x, const int y)		{ return data[width*y + x]; }
	__forceinline ivec4  operator()(const int x, const int y) const { return data[width*y + x]; }
} SourceMap;

//pyramid of coherence maps, keeps track of the mappings at each pyramid level between Ap and Bp
typedef struct SourceMapPyramid {
	SourceMap data[PYRAMID_LEVELS];

	SourceMapPyramid() : data{} {}

	SourceMapPyramid(const int w, const int h)
		: data{}
	{
		int thewidth = w;
		int theheight = h;
		int size = thewidth*theheight;
		data[0].width = thewidth;
		data[0].height = theheight;
		data[0].data.reserve(size);
		data[0].data.resize(size);
		for (int level = 1; level < PYRAMID_LEVELS; ++level) {
			//TODO: downsizing the dimension should be a function
			//divide by powers of two and round
			thewidth = round(thewidth / 2.f);
			theheight = round(theheight / 2.f);
			size = thewidth*theheight;
			data[level].width = thewidth;
			data[level].height = theheight;
			data[level].data.reserve(size);
			data[level].data.resize(size);
		}
	}

	const SourceMapPyramid& operator=(const SourceMapPyramid& rhs) {
		for (int i = 0; i < PYRAMID_LEVELS; ++i) {
			this->data[i] = rhs.data[i];
		}
		return (*this);
	}

	__forceinline SourceMap& operator[](const int level) { return data[level]; }

	__forceinline ivec4& operator()(const int level, const int x, const int y) { return data[level](x, y); }
	__forceinline ivec4  operator()(const int level, const int x, const int y) const { return data[level](x, y); }

} SourceMapPyramid;


typedef struct NV {
	//Neighborhoods vector, Fl(q) in the paper 
	std::vector<float> data[4];
	NV() : data{ std::vector<float>(0), std::vector<float>(0), std::vector<float>(0), std::vector<float>(0) },
		childdim(0), childedge(0), dim(0), edge(0)
	{	}
	// dim is neighborhood size, edge is distance from center pixel to edge of neighborhood 
	// childdim/childedge is for child pyramid level img
	int childdim, childedge, dim, edge;

	void resizeTo(const NV& otherNV) {
		const int size0 = otherNV.data[0].size();
		const int size1 = otherNV.data[1].size();
		const int size2 = otherNV.data[2].size();
		const int size3 = otherNV.data[3].size();
		data[0].reserve(size0);
		data[0].resize(size0);
		data[1].reserve(size1);
		data[1].resize(size1);
		data[2].reserve(size2);
		data[2].resize(size2);
		data[3].reserve(size3);
		data[3].resize(size3);
		childdim = otherNV.childdim;
		childedge = otherNV.childedge;
		dim = otherNV.dim;
		edge = otherNV.edge;
	}

} NV;



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~Stylizer Class, Holder of image pyramids related to ~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~producing B`. Processes A, A`, B data to produce B`~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

class Stylizer {
public:
	Pyramid A, Ap;
	std::vector<Pyramid> B, Bp;
	int numframes;
	pcg32 rng;//.nextFloat() returns value in [0,1)

	//source map, a data struct similar to Pyramid(stores ivec4 instead of features) for the 's' or 'Mt' maps described in the papers
	std::vector<SourceMapPyramid> Smap;
	std::vector<SourceMapPyramid> SmapPM;

public:
	Stylizer();
	~Stylizer();
	Stylizer(const std::string& Apath, const std::string& Aprimepath, const std::string& Bpath, const std::string& outputDir = "", const int numberofframes = 1);

	void GenerateFLOWFeatures();
	void GeneratePyramidsOfB(const std::string& Bpath);
	void RandomizeSourceMapPyramid();
	void InitializePyramidsOfBp(const std::string& outputDir = "");
	void AssignFramePathsToElementsOfB(const std::string& Bpath);
	void RemapLuminance(const int level, const int frame);
	void TestWrite() const;
	void StylizeFramesOfB();
	int CheckPixelOcclusionFwd(const int level, const int frame, const int qx, const int qy);
	int CheckPixelOcclusionBwd(const int level, const int frame, const int qx, const int qy);
	void AdvectFrameForward(const int level, const int frame);
	void AdvectFramesBackward(const int level);
	float GetHeatMapIncrement(const Pyramid& Bpfr, const int level, int& hmlp, int& hmtp, int& hmrp, int& hmbp) const;
	feature GetMatchLocColor(const ivec4& bestmatch, const int level) const;
	void SmapPMInheritResultsFromChild(const int level, const int frame);
	void PrintSmapHeatMap(const int frame) const;
	void WriteAColorMappedLocations() const;
	feature GenFeatureRGBFromYIQ(const ivec4& p, const ivec4& q, const int frame, const int level) const;
	void RotateNeighborhoodIterVector(int& x, int& y, const float sintheta, const float costheta) const;
	void CalcSinCosThetaBetweenNeighborhoodORI(const int level, const int frame, const ivec4& p, const ivec4& q, float& sintheta, float& costheta) const;

	void BestMatch(const int level, const int frame, const ivec4& q, const float appweight, const int iteration, ivec4& bestmatch, feature& TYPE);
	void PatchMatch(const int level, const int frame, const ivec4& q, const NV& BNV, const int iteration, ivec4& patchmatch);
	void ANNMatch(const int level, const int frame, const ivec4& q, const NV& BNV, ivec4& annmatch);
	int CoherenceMatch(const int level, const int frame, const ivec4& q, const NV& BNV, ivec4& cohmatch) const;
	void GenBNeighborhoodsVector_ExtendEdge(const int level, const int frame, const ivec4& q, NV& BNV) const;

	float NeighborhoodsL2Dist_ExtendEdge(const int level, const int frame, const ivec4& p, const ivec4& q, const NV& BNV) const;
	float PixarCoherenceGoal(const int level, const int frame, const ivec4& appmatch, const ivec4& q) const;
	//void ComputeNVBounds(const int level, const int frame, const ivec4& pix, NV& myNV, const NV* BNV = nullptr);
};

Stylizer::Stylizer()
	: A(), Ap(), B(), Bp(), numframes(0), rng(), Smap()
{

}

Stylizer::~Stylizer() {

}

Stylizer::Stylizer(const std::string& Apath, const std::string& Aprimepath, const std::string& Bpath, const std::string& outputDir, const int numberofframes)
	: A(Apath), Ap(Aprimepath), rng(), numframes(numberofframes)
{
	GeneratePyramidsOfB(Bpath);
	RandomizeSourceMapPyramid();
	InitializePyramidsOfBp(outputDir);
	GenerateFLOWFeatures();
	//TestWrite();
	StylizeFramesOfB();
}

void Stylizer::RemapLuminance(const int level, const int frame) {
	// luminance remapping 
	// using sigmaB, sigmaA (standard deviations of luminances of A and B) and muA, muB (mean luminances of A and B) 
	// Y(p) = sigmaB/sigmaA * (Y(p) - muA) + muB
	// std devs and means were calculated during pyramid construction

	const Image& imgB = B[frame][level];
	A[level].RemapLuminance(A[level].stdDevLuminance, imgB.stdDevLuminance, A[level].meanLuminance, imgB.meanLuminance);
	Ap[level].RemapLuminance(A[level].stdDevLuminance, imgB.stdDevLuminance, A[level].meanLuminance, imgB.meanLuminance);
	const int maxpyrlevel = PYRAMID_LEVELS - 1;
	if (level < maxpyrlevel) {//has child, remap the child
		const int childlevel = level + 1;
		const Image& imgBchild = B[frame][childlevel];
		A[childlevel].RemapLuminance(A[childlevel].stdDevLuminance, imgBchild.stdDevLuminance, A[childlevel].meanLuminance, imgBchild.meanLuminance);
		Ap[childlevel].RemapLuminance(A[childlevel].stdDevLuminance, imgBchild.stdDevLuminance, A[childlevel].meanLuminance, imgBchild.meanLuminance);
	}
}

void Stylizer::TestWrite() const {
	cout << "\nA location: " << A[0].filepath;
	cout << "\nAp location: " << Ap[0].filepath;
	cout << "\nB[0] location: " << B[0][0].filepath;
	cout << "\nBp[0] location: " << Bp[0][0].filepath;
	//A.Write(YIQ);
	//Ap.Write(YIQ);
	//B[0].Write(YIQ);
	//Bp[0].Write(YIQ);
	//A.Write(DX2DY2DXY);
	//B[0].Write(DX2DY2DXY);
	//A.Write(ORI);
	//B[0].Write(ORI);
	B[2].Write(FLOW_FWD);
}

void Stylizer::GenerateFLOWFeatures() {
	if (0 == TEMPORALLY_COHERENT) return;

	//FWD
	for (int frame = 0; frame < numframes - 1; ++frame) {
		Pyramid& curr = B[frame];
		const Pyramid* const next = &(B[frame + 1]);

		for (int level = PYRAMID_LEVELS - 1; level >= 0; --level) {
			curr[level].GenFeatureFLOW(&((*next)[level]), level, FLOW_FWD);
		}
	}
	//write 0's to last frame's flow forward features, NEEDED?
	for (int level = PYRAMID_LEVELS - 1; level >= 0; --level) {
		B[numframes - 1][level].GenFeatureFLOW(nullptr, level, FLOW_FWD);
	}

	//BWD
	for (int frame = 1; frame < numframes; ++frame) {
		Pyramid& curr = B[frame];
		const Pyramid* const prev = &(B[frame - 1]);

		for (int level = PYRAMID_LEVELS - 1; level >= 0; --level) {
			curr[level].GenFeatureFLOW(&((*prev)[level]), level, FLOW_BWD);
		}
	}

	//write 0's to first frame's flow backward features, NEEDED?
	for (int level = PYRAMID_LEVELS - 1; level >= 0; --level) {
		B[0][level].GenFeatureFLOW(nullptr, level, FLOW_BWD);
	}
}

void Stylizer::RandomizeSourceMapPyramid() {
	const Image& Bfr0lvl0 = B[0][0];//frame0 level0 
	Smap.reserve(numframes);
	Smap.resize(numframes);
	Smap[0] = SourceMapPyramid(Bfr0lvl0.width, Bfr0lvl0.height);

	for (int level = 0; level < PYRAMID_LEVELS; ++level) {
		const int w = Smap[0][level].width;
		const int h = Smap[0][level].height;

		const int Ap_w = Ap[level].width;
		const int Ap_h = Ap[level].height;

		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				Smap[0](level, x, y) = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
			}
		}
	}//level

	const SourceMapPyramid& refSmap = Smap[0];
	const int smapsize = Smap.size();
	for (int frame = 1; frame < smapsize; ++frame) {
		Smap[frame] = refSmap;
	}
	SmapPM.reserve(numframes);
	SmapPM.resize(numframes);
	for (int frame = 0; frame < smapsize; ++frame) {
		SmapPM[frame] = refSmap;
	}
}

void Stylizer::PrintSmapHeatMap(const int frame) const {
	SourceMapPyramid mysmap = Smap[frame];

	std::string hmfilepath = Bp[0][0].filepath;
	const int hmdotpos = hmfilepath.rfind(".");//rfind searches for last occurance, find searches for first occurance
	const int hmlastFslashpos = hmfilepath.rfind("/", hmdotpos);//second arg tells it to search before that postition
	const std::string hmroot = hmfilepath.substr(0, hmlastFslashpos + 1);//add 1 to include the slash
	const std::string hmfile_ext = hmfilepath.substr(hmdotpos);//from dotpos to null terminator
	std::string heatmapfilepath = hmroot + "RandInitSmap_HEATMAP" + hmfile_ext;
	Pyramid sourceheatmap = Pyramid(heatmapfilepath, Ap[0].width, Ap[0].height, 3, PNG);

	int hmlp, hmtp, hmrp, hmbp;

	const int lastpyrlevel = PYRAMID_LEVELS - 1;
	for (int level = lastpyrlevel; level >= 0; --level) {
		const float increment = GetHeatMapIncrement(Bp[0], level, hmlp, hmtp, hmrp, hmbp);
		const int height = mysmap[level].height;
		const int width = mysmap[level].width;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				ivec4 a = mysmap(level, x, y);
				sourceheatmap(level, RGB, a[0], a[1]) += increment;
			}
		}
	}

	sourceheatmap.Write();
}

void Stylizer::InitializePyramidsOfBp(const std::string& outputDir) {
	this->Bp.reserve(numframes);
	this->Bp.resize(numframes);
	const std::string Bprimepath = this->B[0].pyramid[0].filepath;
	const int dotpos = Bprimepath.rfind(".");//rfind searches for last occurance, find searches for first occurance
	const int lastFslashpos = Bprimepath.rfind("/", dotpos);//second arg tells it to search before that postition
	const std::string root = Bprimepath.substr(0, lastFslashpos + 1);//add 1 to include the slash
	const std::string filename = Bprimepath.substr(lastFslashpos + 1, dotpos - lastFslashpos - 1);
	const std::string file_ext = Bprimepath.substr(dotpos);//from dotpos to null terminator
	const size_t last_letter_index = filename.find_last_not_of("0123456789");
	const std::string filename_stripped = filename.substr(0, last_letter_index + 1);
	const std::string startframenum_str = filename.substr(last_letter_index + 1);
	int startframenum;
	if (startframenum_str == "") {
		startframenum = 0;
	}
	else {
		startframenum = std::stoi(startframenum_str);
	}

	std::string outputdir;
	if (outputDir.length() == 0) {
		//if start from 0 -> startframenum = 0;
		outputdir = root + "STYLIZED_FRAMES/";
		_mkdir(outputdir.c_str());
	}
	else {
		// user passed in an output directory
		outputdir = outputDir + "/";
	}

	const Image& B0_0 = this->B[0].pyramid[0];
	for (int i = 0; i < numframes; ++i) {
		Image& Bi_0 = this->B[i].pyramid[0];
		std::string framename = outputdir + "Bprime" + std::to_string(startframenum + i) + ".png";
		this->Bp[i] = Pyramid(framename, B0_0.width, B0_0.height, B0_0.bpp, B0_0.format);
	}
}

void Stylizer::GeneratePyramidsOfB(const std::string& Bpath) {
	const int dotpos = Bpath.rfind(".");
	const std::string file_ext = Bpath.substr(dotpos);
	bool flag_video = false;
	bool flag_image = false;

	for (int i = 0; i < SUPPORTED_VIDEO_FORMATS.size(); ++i) {
		if (file_ext == SUPPORTED_VIDEO_FORMATS[i]) {
			flag_video = true;
			break;
		}
	}

	if (!flag_video) {
		for (int i = 0; i < SUPPORTED_IMAGE_FORMATS.size(); ++i) {
			if (file_ext == SUPPORTED_IMAGE_FORMATS[i]) {
				flag_image = true;
				break;
			}
		}
	}

	if (flag_video) {
		int lastslashpos;
		const int lastFslashpos = Bpath.rfind("/", dotpos);
		const int lastBslashpos = Bpath.rfind("\\", dotpos);
		if (lastFslashpos > lastBslashpos) {
			lastslashpos = lastFslashpos;
		}
		else {
			lastslashpos = lastBslashpos;
		}
		const std::string root = Bpath.substr(0, lastslashpos + 1);
		const std::string filename = Bpath.substr(lastslashpos + 1, dotpos - lastslashpos - 1);
		const std::string outputdir = root + "FRAMES/";
		_mkdir(outputdir.c_str());//from direct.h (windows), replace later with boost library call to make cross platform
		const std::string outputfileprefix = outputdir + filename + "%d.bmp";//%d needed for frame number
																			 //should not assume 0
		const std::string firstframename = outputdir + filename + "0.bmp";

		//user can pass numframes to be processed
		if (numframes != 0) {
			//TODO: probably need to save the frame rate in the call to SplitVideoIntoFrames 
			//SplitVideoIntoFrames_ffmpegSOMEFRAMES(Bpath, numframes, outputfileprefix);
		}
		else {
			//TODO: probably need to save the frame rate in the call to SplitVideoIntoFrames 
			//numframes = SplitVideoIntoFrames_ffmpegALLFRAMES(Bpath, outputfileprefix);
		}
		AssignFramePathsToElementsOfB(firstframename);
	}
	else if (flag_image) {
		AssignFramePathsToElementsOfB(Bpath);
	}
	else {
		std::cout << "\nERROR: Your edgey file format is not supported.\n";
	}
}

void Stylizer::AssignFramePathsToElementsOfB(const std::string& firstframepath) {
	if (numframes == 1) {
		B.reserve(numframes);
		B.resize(numframes);
		B[0] = Pyramid(firstframepath);
	}
	else {//logic in the else branch assumes trailing frame numbers
		const int dotpos = firstframepath.rfind(".");
		int lastslashpos;
		const int lastFslashpos = firstframepath.rfind("/", dotpos);
		const int lastBslashpos = firstframepath.rfind("\\", dotpos);
		if (lastFslashpos > lastBslashpos) {
			lastslashpos = lastFslashpos;
		}
		else {
			lastslashpos = lastBslashpos;
		}
		const std::string root = firstframepath.substr(0, lastslashpos + 1);
		const std::string filename = firstframepath.substr(lastslashpos + 1, dotpos - lastslashpos - 1);
		const std::string file_ext = firstframepath.substr(dotpos);
		const size_t last_letter_index = filename.find_last_not_of("0123456789");
		const std::string filename_stripped = filename.substr(0, last_letter_index + 1);
		const std::string startframenum_str = filename.substr(last_letter_index + 1);

		if (startframenum_str == "") {
			numframes = 1;
			B.reserve(numframes);
			B.resize(numframes);
			B[0] = Pyramid(firstframepath);
			return;
		}

		const int startframenum = std::stoi(startframenum_str);
		int currentframenum = startframenum;

		//check existance, determine actual frame count that we'll process
		if (numframes == 0) {//must find all
			while (file_exists(root + filename_stripped + std::to_string(++currentframenum) + file_ext)) {}
		}
		else {//user might have asked to render more than actually exist
			const int userlastframe = currentframenum + numframes - 1;
			while (file_exists(root + filename_stripped + std::to_string(++currentframenum) + file_ext) && currentframenum <= userlastframe) {}
		}

		const int finishframenum = currentframenum - 1;
		numframes = currentframenum - startframenum;
		B.reserve(numframes);
		B.resize(numframes);

		for (int i = 0; i < numframes; ++i) {
			const std::string currentfilename = root + filename_stripped + std::to_string(startframenum + i) + file_ext;
			B[i] = Pyramid(currentfilename);
		}
	}
}

//feature Stylizer::GenFeatureRGBFromYIQ(float ApY, const feature& B_YIQ, const ivec4& p, const int level)
feature Stylizer::GenFeatureRGBFromYIQ(const ivec4& p, const ivec4& q, const int frame, const int level) const
{
	// based on the Y channel of Ap and the IQ channels of B, produce RGB of Bp
#if USEAPRIMECOLOR == 1 //scaled back to Ap histogram (normalizing Ap histogram to B (in remapluminance) can make it really dim

	//pulled from remapluminance: solve for Y: float remappedY = ((sigmaB / sigmaA) * (Y - muA)) + muB;
	//const float muA = A[level].meanLuminance;
	//const float muB = B[frame][level].meanLuminance;
	//const float sigmaA = A[level].stdDevLuminance;
	//const float sigmaB = B[frame][level].stdDevLuminance;
	//
	//const float scaledbackY = ((ApY - muB) * (sigmaA / sigmaB)) + muA;
	//const float scaledbackI = B_YIQ[0];
	//const float scaledbackQ = B_YIQ[0];
#if SCALEAPRIMEBACK == 1 
	float r = Ap(level, RGB, p[0], p[1])[0];
	float g = Ap(level, RGB, p[0], p[1])[1];
	float b = Ap(level, RGB, p[0], p[1])[2];
#else
	const float Y = Ap(level, YIQ, p[0], p[1])[0];
	const float I = Ap(level, YIQ, p[0], p[1])[1];
	const float Q = Ap(level, YIQ, p[0], p[1])[2];

	float r = Y + 0.9563 * I + 0.6210 * Q;
	float g = Y - 0.2721 * I - 0.6474 * Q;
	float b = Y - 1.1070 * I + 1.7046 * Q;
#endif

#else //in this case, Ap and B histograms are aligned so there's no scaling back
	const float Y = Ap(level, YIQ, p[0], p[1])[0];
	const float I = B[frame](level, YIQ, q[0], q[1])[1];
	const float Q = B[frame](level, YIQ, q[0], q[1])[2];

	float r = Y + 0.9563 * I + 0.6210 * Q;
	float g = Y - 0.2721 * I - 0.6474 * Q;
	float b = Y - 1.1070 * I + 1.7046 * Q;
#endif
	int clamped = 0;
	clamp(r, 0.f, 1.f, clamped);
	clamp(g, 0.f, 1.f, clamped);
	clamp(b, 0.f, 1.f, clamped);
	return feature(r, g, b);
}

void Stylizer::SmapPMInheritResultsFromChild(const int level, const int frame) {
	if (level == PYRAMID_LEVELS - 1) { return; }//no child

	const int childlevel = level + 1;
	const int mysmapW = SmapPM[0][level].width;
	const int mysmapH = SmapPM[0][level].height;
	for (int y = 0; y < mysmapH; ++y) {
		for (int x = 0; x < mysmapW; ++x) {
			const ivec4 AplocPM = SmapPM[frame](childlevel, x >> 1, y >> 1);
			int axPM = AplocPM[0] << 1;
			int ayPM = AplocPM[1] << 1;
			int clamped = 0;
			clamp(axPM, 0, Ap[level].width - 1, clamped);
			clamp(ayPM, 0, Ap[level].height - 1, clamped);
			SmapPM[frame](level, x, y) = ivec4(axPM, ayPM);
		}
	}

	//inherit Smap child results if on frame 0 
	if (0 == frame) {
		for (int y = 0; y < mysmapH; y++) {
			const bool yodd = (y & 0x0001) == 0x0001;
			for (int x = 0; x < mysmapW; x++) {
				const bool xodd = (x & 0x0001) == 0x0001;
				if (yodd) {
					const ivec4 Aploc = Smap[frame](level, x, y-1);
					int ax = Aploc[0];
					int ay = Aploc[1]+1;
					int clamped = 0; 
					clamp(ax, 0, Ap[level].width - 1, clamped); 
					clamp(ay, 0, Ap[level].height - 1, clamped);
					Smap[frame](level, x, y) = ivec4(ax, ay);
				} else if (!yodd && xodd) {
					const ivec4 Aploc = Smap[frame](level, x-1, y);
					int ax = Aploc[0]+1;
					int ay = Aploc[1];
					int clamped = 0; 
					clamp(ax, 0, Ap[level].width - 1, clamped); 
					clamp(ay, 0, Ap[level].height - 1, clamped);
					Smap[frame](level, x, y) = ivec4(ax, ay);
				} else {
					const ivec4 Aploc = Smap[frame](childlevel, x >> 1, y >> 1);
					int ax = Aploc[0] << 1;
					int ay = Aploc[1] << 1;
					int clamped = 0;
					clamp(ax, 0, Ap[level].width - 1, clamped);
					clamp(ay, 0, Ap[level].height - 1, clamped);
					Smap[frame](level, x, y) = ivec4(ax, ay);
				}
			}//x
		}//y
	}//frame
}

feature Stylizer::GetMatchLocColor(const ivec4& bestmatch, const int level) const {
    feature thecolor((float)bestmatch[0] / (Ap[level].width - 1), 0, (float)bestmatch[1] / (Ap[level].height - 1));

	float bestx = bestmatch[0];
	if (bestx == 0) { thecolor = feature(1, 1, 0); }
	if (bestx == (Ap[level].width - 1)) { thecolor = feature(0, 1, 0); }

	float besty = bestmatch[1];
	if (besty == 0) { thecolor = feature(1,1,1);	}
	if (besty == (Ap[level].height - 1)) { thecolor = feature(0, 1, 1); }
	return thecolor;
}

float Stylizer::GetHeatMapIncrement(const Pyramid& Bpfr, const int level, int& hmlp, int& hmtp, int& hmrp, int& hmbp) const {
	const float BLEFTPIXEL = 0.f;
	const float BTOPPIXEL = 0.f;
	const float BRIGHTPIXEL = Bpfr[0].width - 1;
	const float BBOTPIXEL = Bpfr[0].height - 1;
	hmlp = BLEFTPIXEL / (Bpfr[0].width - 1);//heatmapleftpercent
	hmtp = BTOPPIXEL / (Bpfr[0].height - 1);//heatmaptoppercent
	hmrp = BRIGHTPIXEL / (Bpfr[0].width - 1);//heatmaprightpercent
	hmbp = BBOTPIXEL / (Bpfr[0].height - 1);//heatmapbotpercent
	const int windowwidth = Bpfr[level].width  * (hmrp - hmlp);
	const int windowheight = Bpfr[level].height * (hmbp - hmtp);
	int total = windowheight * windowwidth;
	total = (total * (1 << level) * (1 << level));
	return HEATMAP_SCALE / total;
}

void Stylizer::WriteAColorMappedLocations() const {
	Image ApLOCs = Ap[0];
	std::string filepath = Bp[0][0].filepath;
	const int dotpos = filepath.rfind(".");//rfind searches for last occurance, find searches for first occurance
	const int lastFslashpos = filepath.rfind("/", dotpos);//second arg tells it to search before that postition
	const std::string root = filepath.substr(0, lastFslashpos + 1);//add 1 to include the slash
	const std::string file_ext = filepath.substr(dotpos);//from dotpos to null terminator
	ApLOCs.filepath = root + "ApLOCs" + file_ext;
	ApLOCs.colorByLocation();
	ApLOCs.Write();
}

void Stylizer::StylizeFramesOfB() {
#if WRITE_MATCH_LOC == 1
	WriteAColorMappedLocations();
#endif
#if WRITE_MATCH_LOC == 1 || WRITE_MATCH_TYPE == 1
	std::string hmfilepath = Bp[0][0].filepath;
	const int hmdotpos = hmfilepath.rfind(".");//rfind searches for last occurance, find searches for first occurance
	const int hmlastFslashpos = hmfilepath.rfind("/", hmdotpos);//second arg tells it to search before that postition
	const std::string hmroot = hmfilepath.substr(0, hmlastFslashpos + 1);//add 1 to include the slash
	const std::string hmfile_ext = hmfilepath.substr(hmdotpos);//from dotpos to null terminator
	std::string heatmapfilepath = hmroot + "HEATMAP" + hmfile_ext;
	Pyramid sourceheatmap = Pyramid(heatmapfilepath, Ap[0].width, Ap[0].height, 3, PNG);
#endif


	feature TYPE(0, 0, 0);//keeps track of which match type is being being used, used for debug
	const int lastpyrlevel = PYRAMID_LEVELS - 1;
	for (int level = lastpyrlevel; level >= 0; --level) {//coarse to fine
		for (int frame = 0; frame < numframes; ++frame) {
			RemapLuminance(level, frame);
			SmapPMInheritResultsFromChild(level, frame);
			Pyramid& Bpfr = Bp[frame];
			ivec4 bestmatch(0, 0);
			const int width = Bpfr.pyramid[level].width;
			const int height = Bpfr.pyramid[level].height;
			const float Inv2Pow = 1.f / (1 << level);//divide by powers of 2

													 //IA also mentions weighting cohL2Norm by Inv2Pow?
			const float appweight = 1.f + (KCONST * Inv2Pow);

#if WRITE_MATCH_TYPE == 1 || WRITE_MATCH_LOC == 1
			int hmlp, hmtp, hmrp, hmbp;
			const float increment = GetHeatMapIncrement(Bpfr, level, hmlp, hmtp, hmrp, hmbp);
#endif
			//let patch match converge separately first then incorporate in with coherence on last iter
			int pm;
			for (pm = 0; pm < PATCHMATCH_TOTAL_ITERATIONS - 1; ++pm) {
				for (int qy = 0; qy < height; ++qy) {
					for (int qx = 0; qx < width; ++qx) {
						NV BNV = NV();
						GenBNeighborhoodsVector_ExtendEdge(level, frame, ivec4(qx,qy), BNV);
						PatchMatch(level, frame, ivec4(qx,qy), BNV, pm, ivec4(0,0));
					}
				}
			}

			for (int qy = 0; qy < height; ++qy) {
				for (int qx = 0; qx < width; ++qx) {
					BestMatch(level, frame, ivec4(qx, qy), appweight, pm, bestmatch, TYPE);
					Smap[frame](level, qx, qy) = bestmatch;//needed for coherence match
					Bpfr(level, YIQ, qx, qy) = Ap(level, YIQ, bestmatch[0], bestmatch[1]);//copy entire array of features, not just RGB 
																						  //RENDER RGB in Bprime
#if WRITE_MATCH_TYPE == 1
					Bpfr(level, RGB, qx, qy) = TYPE;
					sourceheatmap(level, RGB, bestmatch[0], bestmatch[1]) += increment;
#elif WRITE_MATCH_LOC == 1
					//Bpfr(level, RGB, qx, qy) = feature((float)bestmatch[0] / (Ap[level].width - 1), 0, (float)bestmatch[1] / (Ap[level].height - 1));
					Bpfr(level, RGB, qx, qy) = GetMatchLocColor(bestmatch, level);
					if ((float)qx / Bpfr[level].width  < hmrp
						&& (float)qx / Bpfr[level].width  > hmlp
						&& (float)qy / Bpfr[level].height > hmtp
						&& (float)qy / Bpfr[level].height < hmbp) {
						sourceheatmap(level, RGB, bestmatch[0], bestmatch[1]) += increment;
					}
	#else
						Bpfr(level, RGB, qx, qy) = GenFeatureRGBFromYIQ(bestmatch, ivec4(qx,qy), frame, level);
	#endif
				}//px
			}//py

#if WRITE_MATCH_LOC == 1 || WRITE_MATCH_TYPE == 1
			sourceheatmap[level].Write();
#endif

			Bp[frame][level].Write();
			std::cout << "\nFrame " << frame << " Level " << level << " written to disk.";


			// call function to advect to next frame
			if (TEMPORALLY_COHERENT == 1 && frame + 1 < numframes) {
				AdvectFrameForward(level, frame);
			}
		}//frame

		if (TEMPORALLY_COHERENT == 1) { AdvectFramesBackward(level); }
	}//level
	PrintSmapHeatMap(0);
}
bool mostlyAlignedInSameDir(const float ax, const float ay, const float bx, const float by) {
	const float lenA = sqrtf(ax*ax + ay*ay);
	const float lenB = sqrtf(bx*bx + by*by);
	const float AdotB = (ax/lenA) * (bx/lenB) + (ay/lenA) * (by/lenB);
	if (AdotB > 0.95) {
		return true;
	} else {
		return false;
	}
}
int Stylizer::CheckPixelOcclusionFwd(const int level, const int frame, const int qx, const int qy) {
	//CHECK IF FLOWS AGREE
	const int width = B[frame][level].width;
	const int height = B[frame][level].height;
	float velFwdX = B[frame](level, FLOW_FWD, qx, qy)[0];
	float velFwdY = B[frame](level, FLOW_FWD, qx, qy)[1];

	int sourceqx = round(qx + velFwdX);
	int sourceqy = round(qy + velFwdY);
	int clamped = 0;
	clamp(sourceqx, 0, width - 1, clamped);
	clamp(sourceqy, 0, height - 1, clamped);
	float velBwd_sourceqx = B[frame + 1][level](FLOW_BWD, sourceqx, sourceqy)[0];
	float velBwd_sourceqy = B[frame + 1][level](FLOW_BWD, sourceqx, sourceqy)[1];
	ivec4 offset(velBwd_sourceqx + velFwdX, velBwd_sourceqy + velFwdY);
	float dist2 = offset[0] * offset[0] + offset[1] * offset[1];
	if (dist2 < OCCLUSION_EPSILON) {
		return 1;
	} else {
		return 0;
	}
}

int Stylizer::CheckPixelOcclusionBwd(const int level, const int frame, const int qx, const int qy) {
	//CHECK IF FLOWS AGREE
	const int width = B[frame][level].width;
	const int height = B[frame][level].height;
	float velBwdX = B[frame](level, FLOW_BWD, qx, qy)[0];
	float velBwdY = B[frame](level, FLOW_BWD, qx, qy)[1];
	int sourceqx = round(qx + velBwdX);
	int sourceqy = round(qy + velBwdY);
	int clamped = 0;
	clamp(sourceqx, 0, width - 1, clamped);
	clamp(sourceqy, 0, height - 1, clamped);
	float velFwd_sourceqx = B[frame - 1](level, FLOW_FWD, sourceqx, sourceqy)[0];
	float velFwd_sourceqy = B[frame - 1](level, FLOW_FWD, sourceqx, sourceqy)[1];
	ivec4 offset(velFwd_sourceqx + velBwdX, velFwd_sourceqy + velBwdY);
	float dist2 = offset[0]*offset[0] + offset[1]*offset[1];
	if (dist2 < OCCLUSION_EPSILON) {
		return 1;
	} else { //pixel is being revealed or hidden
		return 0;
	}
}

void Stylizer::AdvectFrameForward(const int level, const int frame) {
	const int width = Bp[frame].pyramid[level].width;
	const int height = Bp[frame].pyramid[level].height;
	// forward pass
	// for each pixel p: if occluded use PatchMatch result if on coarsest level, else use upsampled result
	// if not occluded, advect offset of previous frame
	for (int qy = 0; qy < height; ++qy) {
		for (int qx = 0; qx < width; ++qx) {
			// if not occluded
			//paper actually says to consult x-(occlbwd) not x+(occlfwd) for frame 
			//if (CheckPixelOcclusionFwd(level, frame, qx, qy) == 0) { continue; }
			if (CheckPixelOcclusionBwd(level, frame + 1, qx, qy) == 0) { continue; }
				//next frame's bwd flow to this pixel is the same as this frames forward flow to that pixel
				//rasterize next frame using previous frame(use FLOW_BWD to find pixel it should use)
				const feature vel = B[frame + 1](level, FLOW_BWD, qx, qy);
				int tmpcohX = round(qx + vel[0]);
				int tmpcohY = round(qy + vel[1]);

				int clamped = 0;
				clamp(tmpcohX, 0, width - 1, clamped);
				clamp(tmpcohY, 0, height - 1, clamped);
				float visibleinboth = 1;
				//if ( (clamped || (qx == tmpcohX && qy == tmpcohY) || frame == 0) ) { 
				//if ( (clamped || frame == 0) ) { 
				if (clamped) { 
					visibleinboth = 0;//want to set to 0 so that we get a fresh render on 0th frame
				}
				//if (clamped) { continue; }
				ivec4 advectedval = Smap[frame](level, tmpcohX, tmpcohY);
				advectedval[2] = visibleinboth;
				Smap[frame + 1](level, qx, qy) = advectedval;
				//Bp[frame+1](level, RGB, qx, qy) = Bp[frame](level, RGB, tmpcohX, tmpcohY);
		}
	}
	//Bp[frame + 1][level].Write();
	//std::cout << "\n Forwards Advected Frame " << frame << " Level " << level << " written to disk.";
}

void Stylizer::AdvectFramesBackward(const int level) {
	for (int frame = numframes - 1; frame > 0; --frame) {
		const int width = Bp[frame][level].width;
		const int height = Bp[frame][level].height;

		// backwards pass: to take the solution of the forward pass in account, randomly choose
		// between the previous solution and the advected result
		for (int qy = 0; qy < height; ++qy) {
			for (int qx = 0; qx < width; ++qx) {
				float rand = rng.nextFloat();
				if (rand >= 0.5) {
					// if not occluded
					//if (CheckPixelOcclusionBwd(level, frame, qx, qy) == 0) { continue; }
					if (CheckPixelOcclusionFwd(level, frame - 1, qx, qy) == 0) { continue; }
						//prev frames fwd flow to this pixel equals this frames bwd flow to that pixel
						//rasterize prev frame using this frame(use FLOW_FWD of prev to find pixel it should use)
						const feature vel = B[frame - 1](level, FLOW_FWD, qx, qy);
						int tmpcohX = round(qx + vel[0]);
						int tmpcohY = round(qy + vel[1]);

						int clamped = 0;
						clamp(tmpcohX, 0, width - 1, clamped);
						clamp(tmpcohY, 0, height - 1, clamped);
						float visibleinboth = 1;
						//if (clamped || (qx == tmpcohX && qy == tmpcohY)) {
						if (clamped) {
							visibleinboth = 0;
						}
						ivec4 advectedval = Smap[frame](level, tmpcohX, tmpcohY);
						advectedval[2] = visibleinboth;
						Smap[frame - 1](level, qx, qy) = advectedval;
						//Bp[frame-1](level, RGB, qx, qy) = Bp[frame](level, RGB, tmpcohX, tmpcohY);
				}
			}
		}

		//std::cout << "\n Backwards Advected Frame " << frame << " Level " << level << " written to disk.";
		Bp[frame - 1][level].Write();
	}
}

void Stylizer::CalcSinCosThetaBetweenNeighborhoodORI(const int level, const int frame, const ivec4& p, const ivec4& q, float& sintheta, float& costheta) const {
#if ROTATE_NEIGHBORHOODS == 1
	const float Abasisx = A(level, ORI, p[0], p[1])[0];
	const float Abasisy = A(level, ORI, p[0], p[1])[1];
	const float Bbasisx = B[frame](level, ORI, q[0], q[1])[0];
	const float Bbasisy = B[frame](level, ORI, q[0], q[1])[1];
	costheta = Abasisx * Bbasisx + Abasisy * Bbasisy;//dot product is cos0
	sintheta = -Abasisy*Bbasisx + Abasisx*Bbasisy;//pulled from last cross product calculation(Z component),in this particular case, it also equals sin0
#endif
}

void Stylizer::RotateNeighborhoodIterVector(int& x, int& y, const float sintheta, const float costheta) const {
#if ROTATE_NEIGHBORHOODS == 1
	x = (int)round(costheta * x + -sintheta*y);
	y = (int)round(sintheta * x +  costheta*y);
#endif
}

void Stylizer::BestMatch(const int level, const int frame, const ivec4& q, const float appweight, const int iteration, ivec4& bestmatch, feature& TYPE) {
	//BNV (B neighborhoods vector) is a concatenation of neighborhood of feature vectors in 
	//B[frame][level+1], Bp[frame][level+1], B[frame][level], Bp[frame][level] centered around qx,qy
	//Do NOT include center pixel qx,qy in Bp[frame][level] 
	NV BNV = NV();
	GenBNeighborhoodsVector_ExtendEdge(level, frame, q, BNV);

	//CohMatch
	ivec4 cohmatch = ivec4();
	float cohL2dist;
	if (q[0] == 0 || q[1] == 0) {//seed border with patchmatchs first, cohmatch needs finished pixels
		cohL2dist = FLT_MAX;
	}
	else if (CoherenceMatch(level, frame, q, BNV, cohmatch) == 1) {//1 indicates that we clipped into the edge of Ap, use patchmatch in this case
		cohL2dist = FLT_MAX;
	}
	else {
		cohL2dist = NeighborhoodsL2Dist_ExtendEdge(level, frame, cohmatch, q, BNV);
		//cohL2dist += PixarCoherenceGoal(level, frame, cohmatch, q);
	}

	//PatchMatch
	ivec4 appmatch = ivec4();
	PatchMatch(level, frame, q, BNV, iteration, appmatch);
	float pmL2dist = NeighborhoodsL2Dist_ExtendEdge(level, frame, appmatch, q, BNV);
	//cohL2dist += PixarCoherenceGoal(level, frame, appmatch, q);

	//AdvectMatch
	ivec4 advmatch = Smap[frame](level, q[0], q[1]);
	float advL2dist;
	if (advmatch[2] == 0) {//if being revealed or covered, render new pixel
		advL2dist = FLT_MAX;
	} else {
		advL2dist = NeighborhoodsL2Dist_ExtendEdge(level, frame, advmatch, q, BNV);
		//advL2dist += PixarCoherenceGoal(level, frame, advmatch, q);
	}

	//InheritMatch (produces lower quality coherence as the frames progress)
	//const bool haschild = level + 1 <= PYRAMID_LEVELS - 1;
	//const bool xodd = (q[0] & 0x0001) == 0x0001;
	//const bool yodd = (q[1] & 0x0001) == 0x0001;
	//if (haschild) {
	//	ivec4 inhmatch;
	//	if(!yodd && !xodd) {
	//	    inhmatch = Smap[frame](level + 1, q[0] >> 1, q[1] >> 1);
	//		inhmatch[0] <<= 1; inhmatch[1] <<= 1;
	//	} else if (yodd) {
	//	    inhmatch = Smap[frame](level, q[0], q[1]-1);
	//		inhmatch[1] += 1;
	//		int clamped = 0;
	//		clamp(inhmatch[1], 0, A[level].height-1, clamped);
	//	} else {
	//	    inhmatch = Smap[frame](level, q[0]-1, q[1]);
	//		inhmatch[0] += 1;
	//		int clamped = 0;
	//		clamp(inhmatch[0], 0, A[level].width-1, clamped);
	//	}
	//	float inhL2dist = NeighborhoodsL2Dist_ExtendEdge(level, frame, inhmatch, q, BNV);
	//    if(inhL2dist < advL2dist && advL2dist != FLT_MAX) {
	//		advL2dist = inhL2dist;
	//		advmatch = inhmatch;
	//    } else if (inhL2dist < cohL2dist) {//if inherited result is better than cohL2dist
	//		cohL2dist = inhL2dist;
	//		cohmatch = inhmatch;
	//	}
	//}

	//penalize certain matches
	pmL2dist *= appweight;//penalize patchmatch, we want the style to be coherent more than we want it to match the shading(patchmatch)
	advL2dist *= ADVECTEDWEIGHT;


	//if (cohL2dist <= pmL2dist && cohL2dist < advL2dist && (advL2dist != FLT_MAX || frame == 0)) {//if advmatch is disoccluded or occluded, (dist ==FLT_MAX) force patchmatch
	if (cohL2dist <= pmL2dist && cohL2dist < advL2dist) {//penalize patchmatch, we want the style to be coherent more than we want it to match the shading(patchmatch)
		bestmatch = cohmatch; bestmatch[3] = 0;
		TYPE = feature(1, 0, 0);
	} else if (pmL2dist < advL2dist) { //coherent match is such a bad match that we'll start seeding with a proper shaded match
		bestmatch = appmatch; bestmatch[3] = 1;
		if (cohL2dist == FLT_MAX) {
			TYPE = feature(1, 1, 1);//white if cohmatch clipped
		} else {
			TYPE = feature(0, 1, 0);
		}
	} else {
		bestmatch = advmatch;
		TYPE = feature(0, 0, 1);
	}
}


void Stylizer::PatchMatch(const int level, const int frame, const ivec4& q, const NV& BNV, const int iteration, ivec4& patchmatch) {
	// random initialization 
	SourceMapPyramid& smap = SmapPM[frame];
	const int Ap_w = Ap[level].width;
	const int Ap_h = Ap[level].height;
	const int smap_w = smap[level].width;
	const int smap_h = smap[level].height;
	const int qx = q[0];
	const int qy = q[1];
	ivec4 random;
	float l2dist_golden;
	const int lastpyrindex = PYRAMID_LEVELS - 1;
	ivec4 goldenP = smap(level, qx, qy);

	for (int i = 0; i < PATCHMATCH_RANDOM_INITIALIZATIONS; ++i) {
		random = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
		float l2dist_scratch = NeighborhoodsL2Dist_ExtendEdge(level, frame, random, q, BNV);

		l2dist_golden = NeighborhoodsL2Dist_ExtendEdge(level, frame, goldenP, q, BNV);

		if (l2dist_scratch <= l2dist_golden) {
			goldenP = random;
			l2dist_golden = l2dist_scratch;
		}
	}

	//propagation: acquire neighbors depending on iteration 
	ivec4 borrowHoriz,borrowVert;
	ivec4 shiftbackHoriz(0,0), shiftbackVert(0,0);//shiftback feature mentioned in patch match seems to be bad idea
	const int ODD = iteration & 0x0001;
	if (ODD == 0x0001) {
	//if (true) {
		if (qx + 1 >= smap_w) {
			borrowHoriz = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
		} else {
			borrowHoriz = smap(level, qx + 1, qy) + ivec4(-1, 0);
			if (borrowHoriz[0] < 0) {
				borrowHoriz = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
			}
		}
		if (qy + 1 >= smap_h) {
			borrowVert = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
		} else {
			borrowVert = smap(level, qx, qy + 1) + ivec4(0,-1);
			if (borrowVert[1] < 0) {
				borrowVert = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
			}
		}
	} else { //very different when finishing on even iterations
		if (qx - 1 < 0) {
			borrowHoriz = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
		} else {
			borrowHoriz = smap(level, qx - 1, qy) + ivec4(1, 0);
			if (borrowHoriz[0] >=  Ap_w) {
				borrowHoriz = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
			}
		}
		if (qy - 1 < 0) {
			borrowVert = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
		} else {
			borrowVert = smap(level, qx, qy - 1) + ivec4(0, 1);
			if (borrowVert[1] >= Ap_h) {
				borrowVert = ivec4(rng.nextFloat() * Ap_w, rng.nextFloat() * Ap_h);
			}
		}
	}

	//check if borrow horizontal or vertical are better matches
	const float l2distVert = NeighborhoodsL2Dist_ExtendEdge(level, frame, borrowVert, q, BNV); //D(f(x+-1,y)) in PatchMatch paper
	const float l2distHoriz = NeighborhoodsL2Dist_ExtendEdge(level, frame, borrowHoriz, q, BNV); // D(f(x,y+-1)) in PatchMatch paper
	if (l2distVert < l2dist_golden && l2distVert <= l2distHoriz) {
		goldenP = borrowVert;
		l2dist_golden = l2distVert;
	} else if (l2distHoriz < l2dist_golden && l2distHoriz < l2distVert) {
		goldenP = borrowHoriz;
		l2dist_golden = l2distHoriz;
	}

	/* random search step */
	ivec4 ui = goldenP;
	int w = max(Ap_w, Ap_h);
	int searchRadius = w >> 1;
	int boundUp, boundDown, boundLeft, boundRight;

	while (searchRadius > 1) {
		boundUp = max(0, ui[1] - searchRadius);
		boundDown = min(Ap_h - 1, ui[1] + searchRadius);
		boundLeft = max(0, ui[0] - searchRadius);
		boundRight = min(Ap_w - 1, ui[0] + searchRadius);

		ivec4 randsearch(rng.nextFloat()*(boundRight - boundLeft + 1) + boundLeft,
			rng.nextFloat()*(boundDown - boundUp + 1) + boundUp);

		float l2distRandom = NeighborhoodsL2Dist_ExtendEdge(level, frame, randsearch, q, BNV);

		if (l2distRandom < l2dist_golden) {
			ui = randsearch;
			l2dist_golden = l2distRandom;
		}
		searchRadius = searchRadius >> 1;
	}
	int clamped = 0;
	clamp(ui[0], 0, Ap_w - 1, clamped);
	clamp(ui[1], 0, Ap_h - 1, clamped);
	smap(level, qx, qy) = ui;
	patchmatch = ui;
}

int Stylizer::CoherenceMatch(const int level, const int frame, const ivec4& q, const NV& BNV, ivec4& cohmatch) const {
	ivec4 rstarCM(-1,0);
	ivec4 rstarPM(-1,0);
	float minl2distCM = FLT_MAX;
	float minl2distPM = FLT_MAX;
	const int edge = BNV.edge;
	const int Bmaxw = B[frame][level].width - 1;
	const int Bmaxh = B[frame][level].height - 1;
	const int Amaxw = A[level].width - 1;
	const int Amaxh = A[level].height - 1;
	bool done = false;
	int clamped = 0;
	const SourceMapPyramid& smapfr = Smap[frame];

	//float sintheta = 0;
	//float costheta = 0;
	//const ivec4 p = smapfr(level, q[0], q[1]);
	//CalcSinCosThetaBetweenNeighborhoodORI(level, frame, p, q, sintheta, costheta);
	for (int y = -edge; y <= edge; ++y) {
		for (int x = -edge; x <= edge; ++x) {
			if (y >= 0 && x >= 0) { done = true; break; }//only want completed pixels (L-shaped neighborhood)  
			ivec4 r(q[0] + x, q[1] + y);
			//RotateNeighborhoodIterVector(r[0], r[1], sintheta, costheta);
			clamp(r[0], 0, Bmaxw, clamped);
			clamp(r[1], 0, Bmaxh, clamped);

			const ivec4 a = smapfr(level, r[0], r[1]);
			ivec4 p = a + (q - r);
			clamped = 0;
			clamp(p[0], 0, Amaxw, clamped);
			clamp(p[1], 0, Amaxh, clamped);

			float l2dist = NeighborhoodsL2Dist_ExtendEdge(level, frame, p, q, BNV);
			if (0 == clamped && l2dist < minl2distCM) {//Best Unclamped 
				minl2distCM = l2dist;
				rstarCM = r;
			} 
			//if (0 == a[3] && 0 == clamped && l2dist < minl2distCM) {//Best Unclamped previously cohmatched CohMatch
			//	minl2distCM = l2dist;
			//	rstarCM = r;
			//} else if (1 == a[3] && 0 == clamped && l2dist < minl2distPM) {//Best Unclamped previoiusly patchmatched CohMatch
			//	minl2distPM = l2dist;
			//	rstarPM = r;
			//}
		}//x
		if (done) { break; }
	}//y

	if (rstarCM[0] != -1) {
		cohmatch = smapfr(level, rstarCM[0], rstarCM[1]) + (q - rstarCM);
		return 0;
	} /*else if (rstarPM[0] != -1) {
		cohmatch = smapfr(level, rstarPM[0], rstarPM[1]) + (q - rstarPM);
		return 0;
	}*/ else {
		return 1;//only getting clamped
	}
}


float Stylizer::NeighborhoodsL2Dist_ExtendEdge(const int level, const int frame, const ivec4& p, const ivec4& q, const NV& BNV) const {
	//Do NOT include center pixel p in the Ap neighborhood feature vectors for Images Ap[level] and Ap[level-1]
	//Throw away comparisons with BNV if A/Ap don't have those neighboring pixels in thier bounds
	float sumofsquareddifferences = 0.f;
	NV diffNV; diffNV.resizeTo(BNV);

	//guassian kernels, mult individual final 3 component sum by squared kernel val?
	const ivec4 childp(p[0] >> 1, p[1] >> 1);
	const ivec4 childq(q[0] >> 1, q[1] >> 1);
	const int childlevel = level + 1;
	const int lastpyrindex = PYRAMID_LEVELS - 1;
	const std::vector<float>& childkern = KERNELS[childlevel];
	const std::vector<float>& kern = KERNELS[level];
	float sintheta = 0;
	float costheta = 0;
	CalcSinCosThetaBetweenNeighborhoodORI(level, frame, p, q, sintheta, costheta);


	//grab offsets and sizes
	const int childdim = diffNV.childdim;
	const int childedge = diffNV.childedge;
	const int dim = diffNV.dim;
	const int edge = diffNV.edge;

	//TAKE DIFFERENCES
	//childA and childB = 0, childAp and childBp = 1 (NV .data indices)
	const int Achildmaxw = A[childlevel].width - 1;
	const int Achildmaxh = A[childlevel].height - 1;
	if (childlevel < PYRAMID_LEVELS) {
		float childsintheta = 0;
		float childcostheta = 0;
		CalcSinCosThetaBetweenNeighborhoodORI(level, frame, childp, childq, childsintheta, childcostheta);
		for (int y = -childedge; y <= childedge; ++y) {
			for (int x = -childedge; x <= childedge; ++x) {
				const int index = (y + childedge)*childdim + (x + childedge);
				int samplex = childp[0] + x;
				int sampley = childp[1] + y;
				RotateNeighborhoodIterVector(samplex, sampley, childsintheta, childcostheta);
				int clamped = 0;
				clamp(samplex, 0, Achildmaxw, clamped);
				clamp(sampley, 0, Achildmaxh, clamped);
				diffNV.data[0][index] = A(childlevel, YIQ, samplex, sampley)[0] - BNV.data[0][index];
				diffNV.data[1][index] = Ap(childlevel, YIQ, samplex, sampley)[0] - BNV.data[1][index];
			}
		}
	}


	//A and B = 2, Ap and Bp = 3 (NV .data indices)
	const int Amaxw = A[level].width - 1;
	const int Amaxh = A[level].height - 1;
	for (int y = -edge; y <= edge; ++y) {
		for (int x = -edge; x <= edge; ++x) {
			const int index = (y + edge)*dim + (x + edge);
			int samplex = p[0] + x;
			int sampley = p[1] + y;
			RotateNeighborhoodIterVector(samplex, sampley, sintheta, costheta);
			int clamped = 0;
			clamp(samplex, 0, Amaxw, clamped);
			clamp(sampley, 0, Amaxh, clamped);
			diffNV.data[2][index] = A(level, YIQ, samplex, sampley)[0] - BNV.data[2][index];
			//if (y <= 0 && x < 0) { //Bpfr neighborhood is L-shaped, and so should Ap's corresponding neighborhood
			if (y <= 0) { //Bpfr neighborhood is L-shaped, and so should Ap's corresponding neighborhood
				//if (y == 0 && x >= 0 && frame == 0) { continue; }
				if (y == 0 && x >= 0) { continue; }
				diffNV.data[3][index] = Ap(level, YIQ, samplex, sampley)[0] - BNV.data[3][index];
			}
		}
	}

	//NORMALIZE the differences?? apply a normalizing weight to the final sum?
	const int totalneighbors = diffNV.data[0].size() + diffNV.data[1].size() + diffNV.data[2].size() + diffNV.data[3].size();
	float numneighborhoods = 4.f;
	if (childlevel > lastpyrindex) {
		numneighborhoods = 2.f;//when there's no child levels to be processed
	}
	const float normalizing_weights[4] = { totalneighbors / (diffNV.data[0].size() * 4.f),
										   totalneighbors / (diffNV.data[1].size() * 4.f),
										   totalneighbors / (diffNV.data[2].size() * numneighborhoods),   //in the case of no children, we'll only be weighting by the neighborhoods in the current level 
										   totalneighbors / (diffNV.data[3].size() * numneighborhoods) };   

	//SQUARE AND SUM THE DIFFERENCES
	//child level
	if (childlevel < PYRAMID_LEVELS) {
		for (int i = 0; i < 2; ++i) {
			const int size = diffNV.data[i].size();
			float sumsqrdiffs_neighborhood = 0.f;//sum for the neighborhood
			for (int j = 0; j < size; ++j) {
				float sumsqrdiffs_pixel = 0.f; //sum for the pixel
				const float featuresdiff = diffNV.data[i][j];
				sumsqrdiffs_pixel += featuresdiff * featuresdiff;
				const float kernelval = childkern[j];
				sumsqrdiffs_neighborhood += (sumsqrdiffs_pixel * kernelval * kernelval);//TODO: does kernel val have to be squared?
			}
			//sumofsquareddifferences += (sumsqrdiffs_neighborhood);// * normalizing_weights[i]);//TODO: does the weight have to be squared?
			sumofsquareddifferences += (sumsqrdiffs_neighborhood * normalizing_weights[i]);//TODO: does the weight have to be squared?
		}
	}
	//current level
	for (int i = 2; i < 4; ++i) {
		const int size = diffNV.data[i].size();
		float sumsqrdiffs_neighborhood = 0.f;//sum for the neighborhood
		for (int j = 0; j < size; ++j) {
			float sumsqrdiffs_pixel = 0.f; //sum for the pixel
			const float featuresdiff = diffNV.data[i][j];
			sumsqrdiffs_pixel += featuresdiff * featuresdiff;
			const float kernelval = kern[j];
			sumsqrdiffs_neighborhood += (sumsqrdiffs_pixel * kernelval * kernelval);//TODO: does the kernelval have to be squared?
		}
		//sumofsquareddifferences += (sumsqrdiffs_neighborhood);// *normalizing_weights[i]);//TODO: does the weight have to be squared?
		sumofsquareddifferences += (sumsqrdiffs_neighborhood * normalizing_weights[i]);//TODO: does the weight have to be squared?
	}

	return sumofsquareddifferences;
}


float Stylizer::PixarCoherenceGoal(const int level, const int frame, const ivec4& p, const ivec4& q) const {
	//does l2 dist on actual source location of pixels from Ap
	//p is a pixel from Ap that potentially will be use. We want a goal that tells us how well that pixel fits into the pixel we are rendering in Bp
	//we take the raster render window in Ap centered at p and diff it with the smap values around q
	//we are not diffing luminance values but summing up the total vector displacements between each position in the two windows

	const std::vector<float>& kern = KERNELS[level];
	float sintheta = 0;
	float costheta = 0;
	CalcSinCosThetaBetweenNeighborhoodORI(level, frame, p, q, sintheta, costheta);

	//grab offsets and sizes
	const int dim = KERNELDIMS[level];
	const int edge = dim >> 1;
	std::vector<float> dists;
	const int dists_size = (dim*dim)/2;
	dists.reserve(dists_size);
	dists.resize(dists_size);

	//A and B = 2, Ap and Bp = 3 (NV .data indices)
	const int Amaxw = A[level].width - 1;
	const int Amaxh = A[level].height - 1;
	const int Bpmaxw = Bp[frame][level].width - 1;
	const int Bpmaxh = Bp[frame][level].height - 1;
	const float PIXAR_RMAX = edge*edge;
	for (int y = -edge; y <= edge; ++y) {
		for (int x = -edge; x <= edge; ++x) {
			const int index = (y + edge)*dim + (x + edge);
			int samplex = p[0] + x;
			int sampley = p[1] + y;
			int Qsamplex = q[0] + x;
			int Qsampley = q[1] + y;
			RotateNeighborhoodIterVector(samplex, sampley, sintheta, costheta);
			int clamped = 0;
			clamp(samplex, 0, Amaxw, clamped);
			clamp(sampley, 0, Amaxh, clamped);
			clamp(Qsamplex, 0, Bpmaxw, clamped);
			clamp(Qsampley, 0, Bpmaxh, clamped);
			//if (y <= 0 && x < 0) { //Bpfr neighborhood is L-shaped, and so should Ap's corresponding neighborhood
			if (y <= 0) { //Bpfr neighborhood is L-shaped, and so should Ap's corresponding neighborhood
				if (y == 0 && x >= 0) { continue; }
				const ivec4 Bpsource = Smap[frame](level, Qsamplex, Qsampley);
				const int xdiff = Bpsource[0] - samplex;
				const int ydiff = Bpsource[1] - sampley;
				float dist = xdiff*xdiff + ydiff*ydiff;
				if (dist > PIXAR_RMAX) dist = PIXAR_RMAX;
				dists[index] = dist;
			}
		}
	}

	float sumsqrdiffs_kernel = 0.f;//sum for the neighborhood
	for (int j = 0; j < dists_size; ++j) {
		float sumsqrdiffs_pixel = 0.f; //sum for the pixel
		const float dist = dists[j];
		const float kernelval = kern[j];
		sumsqrdiffs_kernel += (dist * kernelval * kernelval);//TODO: does the kernelval have to be squared?
	}
	return sumsqrdiffs_kernel;
}

void Stylizer::GenBNeighborhoodsVector_ExtendEdge(const int level, const int frame, const ivec4& q, NV& BNV) const {
	//BNV (B neighborhoods vector) is a concatenation of neighborhood of feature vectors in 
	//B[frame][level+1], Bp[frame][level+1], B[frame][level], Bp[frame][level] centered around corresponding pixels q
	//Do NOT include center pixel qx,qy in Bp[frame][level] and Bp[frame][level+1]
	int childmaxw, childmaxh;
	const int childlevel = level + 1;
	const int maxpyrindex = PYRAMID_LEVELS - 1;
	const Pyramid& Bfr = B[frame];
	const Pyramid& Bpfr = Bp[frame];
	const int maxw = Bfr[level].width - 1;
	const int maxh = Bfr[level].height - 1;

	const int childdim = KERNELDIMS[childlevel];
	const int dim = KERNELDIMS[level];

	if (level == maxpyrindex) {
		childmaxw = 0;
		childmaxh = 0;
	}
	else {
		childmaxw = Bfr[childlevel].width - 1;
		childmaxh = Bfr[childlevel].height - 1;
	}


	const ivec4 childq(q[0] >> 1, q[1] >> 1);
	const int childedge = childdim >> 1;
	const int edge = dim >> 1;

	const int childsize = childdim * childdim;
	const int size = dim      * dim;

	//L-shaped neighborhood in Bp[frame][level]
	const int Lsizex = dim;
	const int Lsizey = edge + 1;
	int Lsize;
	//if (0 == frame) {
	if (true) {
		Lsize = (Lsizex * Lsizey) - (edge + 1);
	} else {
		Lsize = dim * dim;
	}


	BNV.childdim = childdim;
	BNV.childedge = childedge;
	BNV.dim = dim;
	BNV.edge = edge;

	BNV.data[0].reserve(childsize);
	BNV.data[0].resize(childsize);
	BNV.data[1].reserve(childsize);
	BNV.data[1].resize(childsize);
	BNV.data[2].reserve(size);
	BNV.data[2].resize(size);
	BNV.data[3].reserve(Lsize);
	BNV.data[3].resize(Lsize);

	//FILL BNV
	//child Bfr = 0 child Bpfr = 1 (NV .data indices)
	if (childlevel < PYRAMID_LEVELS) {
		for (int y = -childedge; y <= childedge; ++y) {
			for (int x = -childedge; x <= childedge; ++x) {
				const int index = (y + childedge)*childdim + (x + childedge);
				int samplex = childq[0] + x;
				int sampley = childq[1] + y;
				int clamped = 0;
				clamp(samplex, 0, childmaxw, clamped);
				clamp(sampley, 0, childmaxh, clamped);
				BNV.data[0][index] = Bfr(childlevel, YIQ, samplex, sampley)[0];
				BNV.data[1][index] = Bpfr(childlevel, YIQ, samplex, sampley)[0];
			}
		}
	}

	//Bfr = 2 Bpfr = 3 (NV .data indices)
	for (int y = -edge; y <= edge; ++y) {
		for (int x = -edge; x <= edge; ++x) {
			const int index = (y + edge)*dim + (x + edge);
			int samplex = q[0] + x;
			int sampley = q[1] + y;
			int clamped = 0;
			clamp(samplex, 0, maxw, clamped);
			clamp(sampley, 0, maxh, clamped);
			BNV.data[2][index] = Bfr(level, YIQ, samplex, sampley)[0];
			//if (y <= 0 && x < 0) { //Bpfr neighborhood is L-shaped
			if (y <= 0) { //Bpfr neighborhood is L-shaped
				//if (y == 0 && x >= 0 && frame == 0) { continue; }
				if (y == 0 && x >= 0) { continue; }
				BNV.data[3][index] = Bpfr(level, YIQ, samplex, sampley)[0];
			}
		}
	}
}

void Stylizer::ANNMatch(const int level, const int frame, const ivec4& q, const NV& BNV, ivec4& annmatch) {
	float minl2dist = FLT_MAX;
	const int Amaxw = A[level].width;
	const int Amaxh = A[level].height;

	for (int y = 0; y < Amaxh; ++y) {
		for (int x = 0; x < Amaxw; ++x) {
			ivec4 p(x, y);
			float l2dist = NeighborhoodsL2Dist_ExtendEdge(level, frame, p, q, BNV);
			if (l2dist < minl2dist) {
				minl2dist = l2dist;
				annmatch = p;
			}
		}//x
	}//y
}

//void ComputeNVBounds(const int level, const int frame, const ivec4& pixel, NV& myNV, const NV* BNV) {
//	//BNV (B neighborhoods vector) is a concatenation of neighborhood of feature vectors in 
//	//B[frame][level+1], Bp[frame][level+1], B[frame][level], Bp[frame][level] centered around q
//	//Do NOT include center pixel qx,qy in Bp[frame][level] and Bp[frame][level+1]
//
//    //irregular shaped neighborhoods args: const int level, const int frame, const ivec4& q, NV& BNV
//	int childdim,dim;
//	int childmaxw,maxw;
//	int childmaxh,maxh;
//	const int childlevel = level + 1;
//	const Pyramid& Bfr = B[frame];
//	const Pyramid& Bpfr = Bp[frame];
//	
//	const int maxpyrindex = PYRAMID_LEVELS - 1;
//	if (level == maxpyrindex) {
//		childdim = 0;
//		dim = Bfr.kerneldims[0];
//		childmaxw = 0;
//		childmaxh = 0;
//	} else if (childlevel == maxpyrindex) {
//		childdim = Bfr.kerneldims[0];//use largest kernel at the coarsest level to capture low freq components of the image
//		dim = Bfr.kerneldims[level];
//		childmaxw = Bfr[childlevel].width;
//		childmaxh = Bfr[childlevel].height;
//	} else {
//		childdim = Bfr.kerneldims[childlevel];
//		dim = Bfr.kerneldims[level];
//		childmaxw = Bfr[childlevel].width;
//		childmaxh = Bfr[childlevel].height;
//	}
//
//	maxw = Bfr[level].width;
//	maxh = Bfr[level].height;
//
//	const ivec4 childq(q[0] >> 1, q[1] >> 1);
//	const int childedge = childdim >> 1;
//	const int edge      = dim >> 1;
//
//	int childstartx = childq[0] - childedge;
//	int childstopx  = childq[0] + childedge;
//	int childstarty = childq[1] - childedge;
//	int childstopy  = childq[1] + childedge;
//	int startx      = q[0]      - edge;
//	int stopx       = q[0]      + edge;
//	int starty      = q[1]      - edge;
//	int stopy       = q[1]      + edge;
//
//	clamp(childstartx, 0, childmaxw);
//	clamp(childstopx,  0, childmaxw);
//	clamp(childstarty, 0, childmaxh);
//	clamp(childstopy,  0, childmaxh);
//	clamp(startx,      0, maxw);
//	clamp(stopx,       0, maxw);
//	clamp(starty,      0, maxh);
//	clamp(stopy,       0, maxh);
//
//	//convert back to pure offset
//	childstartx -= childq[0];
//	childstopx  -= childq[0];
//	childstarty -= childq[1];
//	childstopy  -= childq[1];
//	startx      -= q[0];
//	stopx       -= q[0];
//	starty      -= q[1];
//	stopy       -= q[1];
//
//	//children are 0 and 1 indices, current level area 2 and 3 indices
//	BNV.startx[0] = childstartx;
//	BNV.startx[1] = childstartx;
//	BNV.startx[2] = startx;
//	BNV.startx[3] = startx;
//	BNV.starty[0] = childstarty;
//	BNV.starty[1] = childstarty;
//	BNV.starty[2] = starty;
//	BNV.starty[3] = starty;
//	BNV.stopx[0]  = childstopx;
//	BNV.stopx[1]  = childstopx;
//	BNV.stopx[2]  = stopx;
//	BNV.stopx[3]  = stopx;
//	BNV.stopy[0]  = childstopy;
//	BNV.stopy[1]  = childstopy;
//	BNV.stopy[2]  = stopy;
//	BNV.stopy[3]  = 0;
//
//	int childsizex = childstopx + childstartx + 1;
//	int childsizey = childstopy + childstarty + 1;
//	int childsize  = childsizex * childsizey;
//	int sizex      = stopx      + startx      + 1;
//	int sizey      = stopy      + starty      + 1;
//	int size       = sizex      * sizey;
//	//L-shaped neighborhood in Bp[frame][level]
//	int Lsizex     = stopx      + startx      + 1;
//	int Lsizey     = 0          + starty      + 1;
//	int Lsize      = (Lsizex * Lsizey) - (stopx + 0 + 1);
//
//	BNV.data[0].reserve(childsize);
//	BNV.data[0].resize(childsize);
//	BNV.data[1].reserve(childsize);
//	BNV.data[1].resize(childsize);
//	BNV.data[2].reserve(size);
//	BNV.data[2].resize(size);
//	BNV.data[3].reserve(Lsize);
//	BNV.data[3].resize(Lsize);
//
//	//FILL BNV
//	//child Bfr = 0
//	//child Bpfr = 1
//	for (int y = childstarty; y <= childstopy; ++y) {
//		for (int x = childstartx; x <= childstopx; ++x) {
//			int index = (y + (-childstarty))*childsizex + (x + (-childstartx));
//			BNV.data[0][index] = Bfr(childlevel, childq[0] + x, childq[1] + y);
//			BNV.data[1][index] = Bpfr(childlevel, childq[0] + x, childq[1] + y);
//		}
//	}
//
//	//Bfr = 2
//	//Bpfr = 3
//	for (int y = starty; y <= stopy; ++y) {
//		for (int x = startx; x <= stopx; ++x) {
//			int index = (y + (-starty))*sizex + (x + (-startx));
//			if (y >= 0 && x >= 0) { //Bpfr neighborhood is L-shaped
//				BNV.data[2][index] = Bfr(level, q[0] + x, q[1] + y);
//			} else {
//				BNV.data[2][index] = Bfr(level, q[0] + x, q[1] + y);
//				BNV.data[3][index] = Bpfr(level, q[0] + x, q[1] + y);
//			}
//		}
//	}
//
 //   //old Stylizer::NeighborhoodsL2Dist(const int level, const ivec4& p, const NV* const BNV) {
	////Do NOT include center pixel p in the Ap neighborhood feature vectors for Images Ap[level] and Ap[level-1]
	////Throw away comparisons with BNV if A/Ap don't have those neighboring pixels in thier bounds
	//float sumofsquareddifferences = 0.f;
	//NV diffNV = NV();
	//diffNV.resizeTo(BNV);
	////ComputeNVBounds(level, 0, p, diffNV, BNV);

////guassian kernels, mult individual final 3 component sum by squared kernel val?
////iterate over diffNV taking differences with 

	////Diff A* and B*
	//const int childlevel = level + 1;
	//const ivec4 childp(p[0] >> 1, p[1] >> 1);
	////grab offsets and sizes
	//const int childstartx = diffNV.startx[0];
	//const int startx      = diffNV.startx[2];
	//const int childstopx  = diffNV.stopx[0];
	//const int stopx       = diffNV.stopx[2]; 
	//const int childsizex  = childstopx + childstartx + 1;
	//const int sizex       = stopx      + startx      + 1;
	//const int childstarty = diffNV.starty[0];
	//const int starty      = diffNV.starty[2];
	//const int childstopy  = diffNV.stopy[0];
	//const int stopy       = diffNV.stopy[2]; 
	//const int childsizey  = childstopy + childstarty + 1;
	//const int sizey       = stopy      + starty      + 1;

////child A and B = 0
////child Ap and Bp = 1
//for (int y = childstarty; y <= childstopy; ++y) {
//	for (int x = childstartx; x <= childstopx; ++x) {
//		int index = (y + (-childstarty))*childsizex + (x + (-childstartx));
//		const int nx = childp[0] + x;
//		const int ny = childp[0] + y;
//		if (nx < 0 || ny < 0 || nx >(A[childlevel].width - 1) || ny >(A[childlevel].height - 1)) {
//			//out of bounds in child A and child Ap, do nothing
//			//diffNV.data[0][index] = features();
//			//diffNV.data[1][index] = features();
//		}
//		diffNV.data[0][index] = A(childlevel, childp[0] + x, childp[1] + y)  - BNV->data[0][index];
//		diffNV.data[1][index] = Ap(childlevel, childp[0] + x, childp[1] + y) - BNV->data[1][index];
//	}
//}

////A and B = 2
////Ap and Bp = 3
//for (int y = starty; y <= stopy; ++y) {
//	for (int x = startx; x <= stopx; ++x) {
//		int index = (y + (-starty))*sizex + (x + (-startx));
//		const int nx = p[0] + x;
//		const int ny = p[0] + y;
//		if (nx < 0 || ny < 0 || nx > (A[level].width - 1) || ny > (A[level].height - 1)) {
//			//out of bounds in child A and child Ap, do nothing
//			//diffNV.data[0][index] = features();
//			//diffNV.data[1][index] = features();
//		} else if (y >= 0 && x >= 0) { //Bpfr neighborhood is L-shaped
//			diffNV.data[2][index] = A(level, p[0] + x, p[1] + y) - BNV->data[2][index];
//		} else {
//			diffNV.data[2][index] = A(level, p[0] + x, p[1] + y)  - BNV->data[2][index];
//			diffNV.data[3][index] = Ap(level, p[0] + x, p[1] + y) - BNV->data[3][index];
//		}
//	}
//}

////NORMALIZE the differences?? apply a weight to the final sum?

////square and sum up all the differences 
//for (int i = 0; i < 4; ++i) {
//	const int size = diffNV.data[i].size();
//	for (int j = 0; j < size; ++j) {
//		const features& featuresdiff = diffNV.data[i][j]; 
//		for (int k = 0; k < NUM_FEATURES; ++k) {
//			//TODO:MULT BY SQUARED KERNEL VAL?
//			float squaredkernelval = 0.f;
//			sumofsquareddifferences += (squaredkernelval * featuresdiff.data[k].l2dist());
//		}
//	}
//}
//return sumofsquareddifferences;
//}

