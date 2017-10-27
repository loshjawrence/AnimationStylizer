#include "AnimationStylizerCmd.h"

#include <maya/MGlobal.h>
#include <maya/MArgList.h>
#include <list>

#include "stylizer.h"

AnimationStylizerCmd::AnimationStylizerCmd() : MPxCommand()
{ 
}


AnimationStylizerCmd::~AnimationStylizerCmd()
{
}

MStatus AnimationStylizerCmd::doIt(const MArgList& args)
{
	MGlobal::displayInfo("Executing Stylization...");
	if (args.length() == 0) {
		MGlobal::displayError("Must specify arguments");
		return MS::kFailure;
	}
	std::string A("C:/Users/grace/Desktop/animationstylizer/ImageTest/ImageTest/gray/A2.png");
	std::string Ap("C:/Users/grace/Desktop/animationstylizer/ImageTest/ImageTest/gray/Ap2.png");
	std::string B("C:/Users/grace/Desktop/animationstylizer/ImageTest/ImageTest/gray/Bface3.png");
	Stylizer(A, Ap, B);
	MGlobal::displayInfo("Finished stylizing! Check the designated output folder.");
	const MStatus failed = MS::kFailure;
	const MStatus success = MS::kSuccess;

	return MStatus::kSuccess;
}

