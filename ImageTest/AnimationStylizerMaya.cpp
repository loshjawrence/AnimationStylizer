#include <maya/MPxCommand.h>
#include <maya/MFnPlugin.h>
#include <maya/MIOStream.h>
#include <maya/MString.h>
#include <maya/MArgList.h>
#include <maya/MGlobal.h>
#include <maya/MSimple.h>
#include <maya/MDoubleArray.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MDGModifier.h>
#include <maya/MPlugArray.h>
#include <maya/MVector.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MStringArray.h>
#include <maya/MFnPlugin.h>
#include <list>

#include "AnimationStylizerCmd.h"

MStatus initializePlugin(MObject obj)
{
	MStatus   status = MStatus::kSuccess;
	MFnPlugin plugin(obj, "MyPlugin", "1.0", "Any");

	// Register Command
	status = plugin.registerCommand("AnimationStylizerCmd", AnimationStylizerCmd::creator);
	if (!status) {
		status.perror("registerCommand");
		return status;
	}

	// Set name
	plugin.setName("Animation Stylizer");

	// Load UI from MEL file
	MGlobal::executeCommand("source \"" + plugin.loadPath() + "/GUI.mel\"");
	status = plugin.registerUI("makeASMenu", "removeASMenu");

	return status;
}

MStatus uninitializePlugin(MObject obj)
{
	MStatus   status = MStatus::kSuccess;
	MFnPlugin plugin(obj);

	// Deregister Command
	//status = plugin.deregisterCommand("LSystemCmd");

	if (!status) {
		status.perror("deregisterCommand");
		return status;
	}
	return status;
}