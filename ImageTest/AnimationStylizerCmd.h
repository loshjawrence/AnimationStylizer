#ifndef CreateAnimationStylizerCmd_H_
#define CreateAnimationStylizerCmd_H_
#include <maya/MPxCommand.h>
#include <string>

class AnimationStylizerCmd : public MPxCommand
{
public:
	AnimationStylizerCmd();
	virtual ~AnimationStylizerCmd();
	static void* creator() { return new AnimationStylizerCmd(); }
	MStatus doIt(const MArgList& args);
};

#endif