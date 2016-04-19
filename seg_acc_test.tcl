proc testSegAcc {} {
	#load image (hard coded for now)
	global gImageVol
	set gImageVol(xml_filename) "/home/gabriel/projects/sample_data/image_data/vti/sample_data-cm.vti"
	createPREOPloadsaveLoadVol

	#load paths
	global gFilenames
	set gFilenames(path_file) "/home/gabriel/research/marsden/tutorial.paths"
	guiFNMloadHandPaths

	#load groups
	set gFilenames(groups_dir) "/home/gabriel/research/marsden/"
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

	global lsGUIcurrentGroup
	global lsGUIcurrentPathNumber
	global lsGUIcurrentPositionNumber

	foreach grpName [group_names] {
		set grpPoints {}
		foreach obj [group_get $grpName] {
			set grpPoints [lappend grpPoints [group_itemid $grpName $obj]]
		}
		puts $grpPoints
		if {[info exists pathmap($grpName)]} {
			set new_name $grpName
			append new_name "_autogen"
			group_create $new_name
			puts $new_name
			puts $pathmap($grpName)

			after 1000

			set lsGUIcurrentGroup $new_name
			set lsGUIcurrentPathNumber $pathmap($grpName)
			set lsGUIcurrentPositionNumber 0
			set itklsGUIParamsBatch(addToGroup) 1

			itkLSDoBatch $pathmap($grpName) $grpPoints $new_name
			#lsGUIaddToGroupBatch "levelset"
			guiSV_group_update_tree
		}
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