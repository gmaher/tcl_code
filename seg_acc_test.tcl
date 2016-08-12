proc runEdgeAnalysis {} {

	global gOptions
	set gOptions(resliceDims) {150 150}



	set imgs [readFromFile "./images.txt"]

	set edge48 [readFromFile "./edge48.txt"]

	set edge96 [readFromFile "./edge96.txt"]

	set grps [readFromFile "./groups.txt"]

	set paths [readFromFile "./paths.txt"]

	set a [llength $imgs]
	set b [llength $edge48]
	set c [llength $edge96]
	set d [llength $grps]
	set e [llength $paths]

	if { $a != $b || $a != $c || $a != $d || $a != $e } {
		error {Length of input files is unequal}
	}

	foreach img $imgs e48 $edge48 e96 $edge96 grp $grps path $paths {
		puts $img
		puts $e48
		puts $e96
		puts $grp
		puts $path
		testSegAcc $img $e48 $path $grp 0 "_image" 2.5 1.5 0.9 0.9
		testSegAcc $img $e48 $path $grp 1 "_edge48" 2.5 1.5 0.9 0.9
		testSegAcc $img $e96 $path $grp 1 "_edge96" 2.5 1.5 0.9 0.9
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

		if {[string match *edge48* $grpName] || [string match *edge96* $grpName] ||
			[string match *image* $grpName]} {
			continue
		}
		#get the points in the existing group
		set grpPoints {}
		foreach obj [group_get $grpName] {
			set grpPoints [lappend grpPoints [group_itemid $grpName $obj]]
		}
		puts $grpPoints

		#See if there is a path that contains the group name
		#if there is then that's probaly the path we are 
		#looking for
		#if there are multiple matches take the shortest one

		#check that a corresponding path exists
		#if so run the level set on all the points
		#store the result in a new group
		set path_to_use 0
		set minLength 1000
		if {[info exists pathmap($grpName)]} {
			set path_to_use $grpName
		} 
		if {$path_to_use == 0} {
			foreach name [array names pathmap] {
				if {[string length $name] < $minLength && [string match *$grpName* $name]} {
					set minLength [string length $name]
					set path_to_use $name
				}
			}
		}
		if {$path_to_use == 0} {
			foreach name [array names pathmap] {
				if {[string length $name] < $minLength && [string match *$name* $grpName]} {
					set minLength [string length $name]
					set path_to_use $name
				}
			}			
		}

		if {$path_to_use != 0} {
			set new_name $grpName
			puts $path_to_use
			append new_name $app

			group_create $new_name
			puts $new_name

			after 1000

			set lsGUIcurrentGroup $new_name
			set lsGUIcurrentPathNumber $pathmap($path_to_use)
			set lsGUIcurrentPositionNumber 0
			set itklsGUIParamsBatch(addToGroup) 1
			set itklsGUIParamsBatch(posList) $grpPoints
			set itklsGUIParams(phyRadius) $rad
			set itklsGUIParams(gSigma1) $blur1
			set itklsGUIParams(gSigma2) $blur2
			set itklsGUIParams(kUpp) $kupp

			lsGUIupdatePath

			itkLSDoBatch_screen $pathmap($path_to_use) $grpPoints $new_name

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

proc readFromFile {fp} {
	set file [open $fp r]
	set data [read $file]
	close $file

	set out [split $data "\n"]

	return $out
} 