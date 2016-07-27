proc runEdgeAnalysis {} {

	global gOptions
	set gOptions(resliceDims) {150 150}

	set imgs(0) "/home/gabriel/projects/tcl_code/models/OSMSC0001/OSMSC0001-cm.mha"
	set imgs(1) "/home/gabriel/projects/tcl_code/models/OSMSC0002/OSMSC0002-cm.mha"
	set imgs(2) "/home/gabriel/projects/tcl_code/models/OSMSC0003/OSMSC0003-cm.mha"
	set imgs(3) "/home/gabriel/projects/tcl_code/models/weiguang/SU0187_2008_247_33758142.mha"
	#set imgs(0) "/home/gabriel/projects/tcl_code/models/OSMSC0006/OSMSC0006-cm.mha"
	

	set edges(0) "/home/gabriel/projects/tcl_code/models/OSMSC0001/OSMSC0001-cm_E.mha"
	set edges(1) "/home/gabriel/projects/tcl_code/models/OSMSC0002/OSMSC0002-cm_E.mha"
	set edges(2) "/home/gabriel/projects/tcl_code/models/OSMSC0003/OSMSC0003-cm_E.mha"
	set edges(3) "/home/gabriel/projects/tcl_code/models/weiguang/SU0187_2008_247_33758142_E.mha" 
	#set edges(0) "/home/gabriel/projects/tcl_code/models/OSMSC0006/OSMSC0006-cm_E48.mha"
	

	set paths(0) "/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/0001_0001-cm.paths"
	set paths(1) "/home/gabriel/projects/tcl_code/models/OSMSC0002/0002_0001/0002_0001-cm.paths"
	set paths(2) "/home/gabriel/projects/tcl_code/models/OSMSC0003/0003_0001/0003_0001-cm.paths"
	set paths(3) "/home/gabriel/projects/tcl_code/models/weiguang/2008_247.paths"
	#set paths(0) "/home/gabriel/projects/tcl_code/models/OSMSC0006/0006_0001/0006_0001-cm.paths"

	set grps(0) "/home/gabriel/projects/tcl_code/models/OSMSC0001/0001_0001/groups"
	set grps(1) "/home/gabriel/projects/tcl_code/models/OSMSC0002/0002_0001/groups"
	set grps(2) "/home/gabriel/projects/tcl_code/models/OSMSC0003/0003_0001/groups"
	set grps(3) "/home/gabriel/projects/tcl_code/models/weiguang/groups"
	#set grps(0) "/home/gabriel/projects/tcl_code/models/OSMSC0006/0006_0001/groups"


	for {set index 0} {$index < 4} {incr index} {
		puts $imgs($index)
		testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 0 "_edge48" 2.5 1.5 0.9 0.9
		testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 1 "_image48" 2.5 1.5 0.9 0.9
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 0 "_edge_bl05" 0.5 1.5 0.9 0.9 
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 1 "_image_bl05" 0.5 1.5 0.9 0.9
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 0 "_edge_bl01_r015_ku25" 0.1 1.5 0.15 2.5 
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 1 "_image_bl01_r015_ku25" 0.1 1.5 0.15 2.5
	}
 
}

proc testSegAcc {imgName edgeName pathName grpName use_edge app blur1 blur2 rad kupp} {
	#load image (hard coded for now)
	global gImageVol
	#set gImageVol(xml_filename) "/home/gabriel/projects/sample_data/image_data/vti/sample_data-cm.vti"
	#set gImageVol(mha_filename) $imgName

	if {[string match *.mha* $imgName]} {
		set gImageVol(mha_filename) $imgName
	} else {
		set gImageVol(xml_filename) $imgName
	}

	if {$use_edge == 1} {
		seg_LoadEdgeImageMha $edgeName
	}
	createPREOPloadsaveLoadVol

	#load paths
	global gFilenames
	#set gFilenames(path_file) "/home/gabriel/research/marsden/tutorial.paths"
	set gFilenames(path_file) $pathName
	guiFNMloadHandPaths

	#load groups
	#set gFilenames(groups_dir) "/home/gabriel/research/marsden/"
	set gFilenames(groups_dir) $grpName
	createPREOPgrpLoadGroups
	guiSV_group_update_tree

	after 1000

	#get path names
	global gPathPoints
	set pathids {}
	set pathnames {}

	foreach i [array names gPathPoints] {
		set ab [split $i ,]
		set a [lindex $ab 0]
		set b [lindex $ab 1]
		if {$b == "name"} {
			set pathids [lappend pathids $a]
			set pathnames [lappend pathnames $gPathPoints($i)]
			set pathmap($gPathPoints($i)) $a
		}
	}

	#pathids now contains all pathids
	#pathnames now contains all path names
	#pathmap maps from pathnames to ids
	puts $pathids
	puts $pathnames
	puts [array names pathmap]

	after 1000

	global lsGUIcurrentGroup
	global lsGUIcurrentPathNumber
	global lsGUIcurrentPositionNumber
	global itklsGUIParamsBatch
	global itklsGUIParams
	global symbolicName

	if {$use_edge == 1} {
		set itklsGUIParams(2DEdgeImage) "LSEdge"
		set itklsGUIParams(useEdgeImage) "disp"
	} else {
		set itklsGUIParams(useEdgeImage) 0
	}

	foreach grpName [group_names] {
		#get the points in the existing group
		set grpPoints {}
		foreach obj [group_get $grpName] {
			set grpPoints [lappend grpPoints [group_itemid $grpName $obj]]
		}
		puts $grpPoints

		#check that a corresponding path exists
		#if so run the level set on all the points
		#store the result in a new group
		if {[info exists pathmap($grpName)]} {
			set new_name $grpName

			append new_name $app

			group_create $new_name
			puts $new_name
			puts $pathmap($grpName)

			after 1000

			set lsGUIcurrentGroup $new_name
			set lsGUIcurrentPathNumber $pathmap($grpName)
			set lsGUIcurrentPositionNumber 0
			set itklsGUIParamsBatch(addToGroup) 1
			set itklsGUIParamsBatch(posList) $grpPoints
			set itklsGUIParams(phyRadius) $rad
			set itklsGUIParams(gSigma1) $blur1
			set itklsGUIParams(gSigma2) $blur2
			set itklsGUIParams(kUpp) $kupp

			lsGUIupdatePath

			itkLSDoBatch_screen $pathmap($grpName) $grpPoints $new_name

			guiSV_group_update_tree
		}
	}
	guiSV_group_update_tree

	createPREOPgrpSaveGroups

	#delete groups so they don't carry over to next image
	foreach grp [group_names] {
		group_delete $grp
	}

	guiSV_group_update_tree
}

#NOTES:
#guiCVloadVTI
#guiCVloadMha -> sets global gImageVol(mha_filename) to image filename
#guiPPloadPaths -> sets global gImageVol(path_file) to path filename, calls guiFNMloadHandPaths
#guiSV_group_load_groups -> sets directory, calls guiSV_group_update_tree, can also get group names
# by doing [group_names]
# can get group ids by doing [group_iditems groupName {}]
#guiSV_model_load_model
#guiFNMloadHandPaths -> calls guiSV_path_update_tree
#gPathPoints: global variable that contains labels (pathID, pointID) and (pathID, "name") and (int, splintPts)
#the path data can be accessed via gPathPoint(pathId,pointId)
#lsGUIaddToGroup {type}, set type = "levelset"
#createPREOPgrpLoadGroups
#guiSV_group_save_groups: saves groups by calling
#createPREOPgrpSaveGroups, saves every group in group_names by calling
#group_saveProfiles

#Button binds to guiSV_group_new_group (use this to create group?)
#runs group_create $name and guiSV_group_update_tree (this creates a group)

##ADDING TO GROUP/RUNNING LEVELSET
#lsGUIaddToGroup{"levelset"} looks at lsGUIcurrentPositionNumber, 
#lsGUIcurrentPathNumber, lsGUIcurrentGroup
#levelset runs separately from add to group

#For batch level set
#need phyRadius set and need to update path start and end points

proc itkLSDoBatch_screen {pathId posList groupName} {

	# set base string
	global lsGUIcurrentPositionNumber
	global lsGUIcurrentPathNumber
	global lsGUIcurrentGroup
	global gOptions
	global gPathPoints

	set orgPosId $lsGUIcurrentPositionNumber
	set orgPathId $lsGUIcurrentPathNumber
	set orgGroup $lsGUIcurrentGroup
	set lsGUIcurrentPathNumber $pathId
	set lsGUIcurrentGroup $groupName
	
	global itklsGUIParamsBatch
	set addToGroup $itklsGUIParamsBatch(addToGroup)
	set smooth $itklsGUIParamsBatch(smooth)

	set pll [llength $posList]
	#puts "pathId: $pathId \n"
	set notDoneList {}
	for {set idx 0} {$idx < [llength $posList]} {incr idx 1} {
		set lsGUIcurrentGroup $groupName
		
		set posId [lindex $posList $idx]
		global lsGUIcurrentPositionNumber
		set lsGUIcurrentPositionNumber $posId

		itkLSOnPos $pathId $posId
		set seg /lsGUI/$pathId/$posId/ls/oriented

		set addToGrp $itklsGUIParamsBatch(addToGroup)
		if { $itklsGUIParamsBatch(smooth) == "1" } {
		    if { [lsGUIfourierSmoothSeg itklevelset 1] != "0"} {
		    	set addToGrp "0"
		    }
		}

		catch {repos_delete -obj tmp/pd}
		if { [catch { geom_sampleLoop -src $seg -num 20 -dst tmp/pd} fid] } {
			puts "Cannot loft:\n$fid"
			set addToGrp "0"
			lappend notDoneList $posId
			catch {repos_delete -obj tmp/pd}
		}
		catch {repos_delete -obj tmp/pd}

	
		if { $addToGrp == "1" } {
			lsGUIaddToGroup levelset
		}	
		after 1

		#vis_renWriteJPEG lsRenWinPot_ren1 ./screens/$groupName.$pathId.$posId

		lsGUIupdatePositionScale 0
		itklsChangeFrame 0

	}

puts "notDoneList $notDoneList"

} 