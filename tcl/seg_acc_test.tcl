proc runGroupsToVTKonFiles {imgs edges paths groups edge_string use_edge} {
	#Runs batch level set for models in a list of files
	#
	#args:
	#	@a: imgs - list of img pathnames
	#	@a edges - list of edge maps
	#	@a paths - list of path files
	# @a grps - list of group folders
	#	@a img_string, string, string to append to groups generated using
	#	levelset without an edgemap
	# @a edge_string, string, string to append to groups generated using
	#	edgemap
	# @a use_edge, int, 0 or 1 whether to use the edgemap or not
	#
	#preconditions:
	#	*@a all files have the same number of lines

	global gOptions
	set gOptions(resliceDims) {150 150}

	set imgs [readFromFile $imgs]
	set edges [readFromFile $edges]
	set paths [readFromFile $paths]
	set groups [readFromFile $groups]

	foreach I $imgs E $edges P $paths G $groups {
		puts "$I\n $E\n $P\n $G\n"


	}
}

proc runEdgeOnFiles {imgs edges paths groups img_string edge_string use_edge} {
	#Runs batch level set for models in a list of files
	#
	#args:
	#	@a: imgs - list of img pathnames
	#	@a edges - list of edge maps
	#	@a paths - list of path files
	# @a grps - list of group folders
	#	@a img_string, string, string to append to groups generated using
	#	levelset without an edgemap
	# @a edge_string, string, string to append to groups generated using
	#	edgemap
	# @a use_edge, int, 0 or 1 whether to use the edgemap or not
	#
	#preconditions:
	#	*@a all files have the same number of lines

	global gOptions
	set gOptions(resliceDims) {150 150}

	set imgs [readFromFile $imgs]
	set edges [readFromFile $edges]
	set paths [readFromFile $paths]
	set groups [readFromFile $groups]

	foreach I $imgs E $edges P $paths G $groups {
		puts "$I\n $E\n $P\n $G\n"

		if {[checkEdgeGroupExists $G $edge_string]} {
			puts "found existing edge/image groups, continuing"
			continue
		}

		testSegAcc $I $E $P $G $use_edge $edge_string 2.5 1.5 0.3 0.9
	}
}

proc runEdgeAnalysis {vasc_dir edge_code img_string edge_string} {
	#Runs batch level set for every model in the vascular data repository
	#that satisfies certain conditions. Groups are computed both without and
	#with an edgemap
	#
	#args:
	#	@a vasc_dir, string, path to the vascular_data repository folder
	#	@a edge_code, string,  pattern to look for in
	#	edgemap file names. e.g. if OSMSC0001-cm_E96.mha is the edgemap
	#	filename then a suitable code would be _E96.mha
	#	@a img_string, string, string to append to groups generated using
	#	levelset without an edgemap
	# 	@a edge_string, string, string to append to groups generated using
	#	edgemap
	#
	#preconditions:
	#	*@a edge_code only mathces one file in each directory in the vascular
	#	data repository

	global gOptions
	set gOptions(resliceDims) {150 150}

	set dirs [glob -types {d} $vasc_dir/OSMSC*]

	foreach dir $dirs {
		set edge [glob -nocomplain $dir/*$edge_code]
		set path [glob -nocomplain $dir/*/*.paths]
		set grp [glob -nocomplain $dir/*/*_groups-cm]
		set img [glob -nocomplain $dir/*OSMSC*-cm.mha]

		puts "$img $edge $path $grp"

		if {[llength $grp] != 1 || [llength $path] != 1 || [llength $edge] != 1 ||
			[llength $img] != 1} {
				puts "More than 1 file or folder (or no file or folder) matches regular expressions, continuing"
				continue
		}

		if {[checkEdgeGroupExists $grp $edge_string]} {
			puts "found existing edge/image groups, continuing"
			continue
		}

		testSegAcc $img $edge $path $grp 0 $img_string 2.5 1.5 0.3 0.9
		testSegAcc $img $edge $path $grp 1 $edge_string 2.5 1.5 0.3 0.9

		#take screenshots
		takeScreenshots $img $edge $path $grp $dir
	}


		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 0 "_edge_bl05" 0.5 1.5 0.9 0.9
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 1 "_image_bl05" 0.5 1.5 0.9 0.9
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 0 "_edge_bl01_r015_ku25" 0.1 1.5 0.15 2.5
		#testSegAcc $imgs($index) $edges($index) $paths($index) $grps($index) 1 "_image_bl01_r015_ku25" 0.1 1.5 0.15 2.5
}

proc runScreenshots {vasc_dir edge_code estring} {
	#Runs batch level set for every model in the vascular data repository
	#that satisfies certain conditions. Groups are computed both without and
	#with an edgemap
	#
	#args:
	#	@a vasc_dir, string, path to the vascular_data repository folder
	#	@a edge_code, string,  pattern to look for in
	#	edgemap file names. e.g. if OSMSC0001-cm_E96.mha is the edgemap
	#	filename then a suitable code would be _E96.mha
	#	@a estring, string, string to append to directory containing edgemap
	# 	screenshots

	global gOptions
	set gOptions(resliceDims) {150 150}

	set dirs [glob -types {d} $vasc_dir/OSMSC*]

	foreach dir $dirs {
		set edge [glob -nocomplain $dir/*$edge_code]
		set path [glob -nocomplain $dir/*/*.paths]
		set grp [glob -nocomplain $dir/*/*_groups-cm]
		set img [glob -nocomplain $dir/*OSMSC*-cm.mha]

		puts "$img $edge $path $grp"

		if {[llength $grp] != 1 || [llength $path] != 1 || [llength $edge] != 1 ||
			[llength $img] != 1} {
				puts "More than 1 file or folder (or no file or folder) matches regular expressions, continuing"
				continue
		}

		takeScreenshots $img $edge $path $grp $dir $estring
	}
}

proc load_svfiles {imgName pathName grpName} {
	global gImageVol

	if {[string match *.mha* $imgName]} {
		set gImageVol(mha_filename) $imgName
	} else {
		set gImageVol(xml_filename) $imgName
	}

	seg_LoadEdgeImageMha $edgeName

	createPREOPloadsaveLoadVol

	#load paths
	global gFilenames
	set gFilenames(path_file) $pathName
	guiFNMloadHandPaths

	#load groups
	set gFilenames(groups_dir) $grpName
	createPREOPgrpLoadGroups
	guiSV_group_update_tree
}

proc groupsToVTK {imgName edgeName pathName grpName} {
	global gImageVol
	global gFilenames
	load_svfiles imgName pathName grpName

	if {$use_edge == 1} {
		seg_LoadEdgeImageMha $edgeName
	}
	createPREOPloadsaveLoadVol

	foreach grpName [group_names] {

	}
}

proc takeScreenshots {imgName edgeName pathName grpName dir estring} {

	set edge_dir $dir/screens/edge
	set img_dir $dir/screens/img

	global gImageVol
	global gFilenames
	load_svfiles imgName pathName grpName

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


	set itklsGUIParams(2DEdgeImage) "LSEdge"
	set itklsGUIParams(useEdgeImage) "disp"


	foreach grpName [group_names] {

		#get the points in the existing group
		set grpPoints {}
		foreach obj [group_get $grpName] {
			set grpPoints [lappend grpPoints [group_itemid $grpName $obj]]
		}
		puts $grpPoints

		if {[info exists pathmap($grpName)]} {

			set pathid $pathmap($grpName)
			set numPathPts $gPathPoints($pathid,numSplinePts)
			set grpEnd [llength $grpPoints]
			set maxGrpPt [lindex $grpPoints [expr $grpEnd-1]]

			if {$maxGrpPt > [expr $numPathPts]} {
				puts "path and group do not have same number of points, continuing"
				continue
			}

			seg_writeSliceTiff $pathmap($grpName) $grpPoints volume_image $img_dir/$grpName
			seg_writeSliceTiff $pathmap($grpName) $grpPoints $itklsGUIParams(edgeImage) $edge_dir/$estring/$grpName

		}


	}

	#delete groups so they don't carry over to next image
	foreach grp [group_names] {
		group_delete $grp
	}

	guiSV_group_update_tree
}

proc testSegAcc {imgName edgeName pathName grpName use_edge app blur1 blur2 rad kupp} {

	global gImageVol
	global gFilenames
	load_svfiles imgName pathName grpName

	if {$use_edge == 1} {
		seg_LoadEdgeImageMha $edgeName
	}
	createPREOPloadsaveLoadVol

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
			[string match *image* $grpName] || [string match *$app* $grpName]} {
			continue
		}
		#get the points in the existing group
		set grpPoints {}
		foreach obj [group_get $grpName] {
			set grpPoints [lappend grpPoints [group_itemid $grpName $obj]]
		}
		puts $grpPoints

		#check that a corresponding path exists
		#if so run the level set on all the points
		#store the result in a new group
		set path_to_use 0
		set minLength 1000
		if {[info exists pathmap($grpName)]} {
			set path_to_use $grpName
		}

		if {$path_to_use != 0} {
			set new_name $grpName
			puts $path_to_use
			append new_name $app

			group_create $new_name
			puts $new_name

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

proc checkEdgeGroupExists {grp_dir edge_code} {
	set files [glob $grp_dir/*]

	foreach file $files {
		if {[string match *$edge_code* $file]} {
			return 1
		}
	}

	return 0
}
