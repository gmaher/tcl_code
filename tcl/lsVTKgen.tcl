proc runGroupsToVTKonFiles {imgs paths groups} {
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

	#global gOptions
	#set gOptions(resliceDims) {64 64}

	set imgs [readFromFile $imgs]
	set paths [readFromFile $paths]
	set groups [readFromFile $groups]

	foreach I $imgs P $paths G $groups {
		puts "$I\n $P\n $G\n"
    generate_truth_groups $I $P $G

	}
}

proc run_params {params img path grp edge edgeString imageString edgeType} {
  #Runs group generation with and without edge map for different
  #level set parameters
	#params(0) = phyRadius
	#params(1) = kLow
	#params(2) = kThr
	#params(3) = kUpp
	#

  global itklsGUIParams
	set itklsGUIParams(phyRadius) [lindex $params 0]
	set itklsGUIParams(kLow) [lindex $params 1]
	set itklsGUIParams(kThr) [lindex $params 2]
	set itklsGUIParams(kUpp) [lindex $params 3]

  set edge_code $edgeString
  set img_code $imageString

  #generate image groups
  generate_edge_groups $img $path $grp 0 $img_code
  close_files

  #generate edge groups
  generate_edge_groups $img $path $grp $edge $edge_code $edgeType
  close_files

}

# Code for generating level set segmentations for the cardiovascular
# dataset, and for extracting the user generated groups as vtk contours.
# Additionally the potential and gradient magnitude windows can be outputted.
#
#
proc run_kthr {kthr_list img path grp edge edgeString} {
  #Runs group generation with and without edge map for different
  #kthr values

  global itklsGUIParams
  foreach kthr $kthr_list {
    set itklsGUIParams(kThr) $kthr
    set edge_code ${edgeString}_kthr${kthr}
    set img_code image_kthr${kthr}

    #generate image groups
    generate_edge_groups $img $path $grp 0 $img_code
    close_files

    #generate edge groups
    generate_edge_groups $img $path $grp $edge $edge_code
    close_files
  }
}

proc generate_truth_groups {img path grp} {
  # Function that takes in image, path and group file paths
  # reorients the groups
  # then loops over every group for which there is a corresponding path
  # and outputs the segmentation at every group member location
  # the mag and pot windows are also saved
  #
  # all files are saved to the current directory
  #
  # args:
  #   @a img: string, the path to the image to load
  #   @a path: string, the path to the path file to load
  #   @a grp: string, the path to the group folder to load
  puts "loading files"
  global gImageVol
  global gFilenames
  load_files $img $path $grp

  puts "loading pathmap"
  global pathmap
  get_path_map

  puts "getting image name"
  set imgname [get_image_name $img]

  puts "starting group loop"
  foreach grp [array names pathmap] {
		puts $grp
		if {[group_exists $grp]} {
			set pathid $pathmap($grp)
	    group_restorePreopSegs $grp

	    global grpPoints
	    get_group_points $grp

	    puts "starting point loop"
	    foreach point $grpPoints {
	      puts "$imgname $pathid $grp $point"
	      lsGUIMake2DImages $pathid $point
	      set ls_fn ${imgname}.${grp}.${point}.truth.ls.vtp
	      set mag_fn ${imgname}.${grp}.${point}.truth.mag.vts
	      set pot_fn ${imgname}.${grp}.${point}.truth.pot.vts
				if {[repos_exists -obj /lsGUI/$pathid/$point/thr/selected]} {
	      	repos_writeVtkPolyData -file $ls_fn -obj /lsGUI/$pathid/$point/thr/selected -type ascii
	      	repos_writeVtkStructuredPoints -file $mag_fn -obj /tmp/lsGUI/mag -type ascii
	      	repos_writeVtkStructuredPoints -file $pot_fn -obj /tmp/lsGUI/pot -type ascii
				}
	    }
		}
  }
	puts "closing files"
  close_files

}

proc generate_edge_groups {img path grp {edge 0} {edgeString image} {edgeType "userEdge"}} {
  # Function that takes in image, path, group and edge file paths
  # loops over every group for which there is a corresponding path
  # computes level set at each group member location
  # outputs the level set, the gradient magnitude and the potential
  # all files are saved to the current directory
  #
  # args:
  #   @a img: string, the path to the image to load
  #   @a path: string, the path to the path file to load
  #   @a grp: string, the path to the group folder to load
  #   @a edge: string, the path to the edge map to load
  #   @a edgeString: string, code to append to the outputted files

  puts "loading files"
  global gImageVol
  global gFilenames
  load_files $img $path $grp $edge

  global itklsGUIParams

  if {$edge ne 0} {
		set itklsGUIParams(2DEdgeImage) $edgeType
		set itklsGUIParams(useEdgeImage) "disp"
	} else {
		set itklsGUIParams(useEdgeImage) 0
	}

  puts "loading pathmap"
  global pathmap
  get_path_map

  puts "getting image name"
  set imgname [get_image_name $img]

  #Also need to set some global parameters for level set
  global lsGUIcurrentPositionNumber
	global lsGUIcurrentPathNumber
	global lsGUIcurrentGroup
	global gOptions
	global gPathPoints
  global itklsGUIParams

  ####################################################
  # Start running level set
  ####################################################
  puts "starting group loop"
  foreach grp [array names pathmap] {
    set pathid $pathmap($grp)
    group_restorePreopSegs $grp

    global grpPoints
    get_group_points $grp

    puts "starting point loop"
    foreach point $grpPoints {
      puts "$imgname $grp $pathid $point"

      set lsGUIcurrentPathNumber $pathid
      set lsGUIcurrentGroup $grp
      set lsGUIcurrentPositionNumber $point

      itkLSOnPos $pathid $point

      set ls_fn ${imgname}.${grp}.${point}.${edgeString}.ls.vtp
      set mag_fn ${imgname}.${grp}.${point}.${edgeString}.mag.vts
      set pot_fn ${imgname}.${grp}.${point}.${edgeString}.pot.vts

      repos_writeVtkPolyData -file $ls_fn -obj /lsGUI/$pathid/$point/ls -type ascii
      repos_writeVtkStructuredPoints -file $mag_fn -obj /img/$pathid/$point/mag -type ascii
      repos_writeVtkStructuredPoints -file $pot_fn -obj /img/$pathid/$point/pot -type ascii

    }
  }
}

proc get_image_name {fn} {
  #function to parse the filepath to an image to get only the image name
  #
  #args:
  #   @a fn - string, the filepath to the image
  set f [lindex [split $fn /] end]
  string map {-cm.mha ""} $f
}

proc get_group_points {grpName} {
  #function to get the points of a group
  #
  #note requires calling global grpPoints before calling this procedure
  global grpPoints
  set grpPoints {}
  foreach obj [group_get $grpName] {
    set grpPoints [lappend grpPoints [group_itemid $grpName $obj]]
  }
}

proc get_path_map {} {
  #function to get a list that maps pathids to group names
  #
  #Note requires calling global pathmap before calling this procedure
	global gPathPoints
  global pathmap
  array set pathmap {}
  unset pathmap
  array set pathmap {}
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
}

proc load_files {img path grp {edge 0}} {
  global gImageVol

  if {[string match *.mha* $img]} {
    set gImageVol(mha_filename) $img
  } else {
    set gImageVol(xml_filename) $img
  }

  if {$edge ne "0"} {
    seg_LoadEdgeImageMha $edge
  }
  createPREOPloadsaveLoadVol

  #load paths
  global gFilenames
  set gFilenames(path_file) $path
  guiFNMloadHandPaths

  #load groups
  set gFilenames(groups_dir) $grp
  createPREOPgrpLoadGroups
  guiSV_group_update_tree
}

proc close_files {} {
  #delete groups so they don't carry over to next image
  foreach grp [group_names] {
    group_delete $grp
  }

  guiSV_group_update_tree
}

proc readFromFile {fp} {
	set file [open $fp r]
	set data [read $file]
	close $file

	set out [split $data "\n"]

	return $out
}
