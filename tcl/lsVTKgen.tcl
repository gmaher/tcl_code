# Code for generating level set segmentations for the cardiovascular
# dataset, and for extracting the user generated groups as vtk contours.
# Additionally the potential and gradient magnitude windows can be outputted.
#
#
proc run {imgFileList pathFileList grpFileList } {

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
    set pathid $pathmap($grp)
    group_restorePreopSegs $grp

    global grpPoints
    get_group_points $grp

    puts "starting point loop"
    foreach point $grpPoints {
      puts "$imgname $grp $point"
      lsGUIMake2DImages $pathid $point
      set ls_fn ${imgname}.${grp}.${point}.ls.truth.vtp
      set mag_fn ${imgname}.${grp}.${point}.mag.truth.vts
      set pot_fn ${imgname}.${grp}.${point}.pot_truth.vts

      repos_writeVtkPolyData -file $ls_fn -obj /lsGUI/$pathid/$point/thr/selected -type ascii
      repos_writeVtkStructuredPoints -file $mag_fn -obj /tmp/lsGUI/mag -type ascii
      repos_writeVtkStructuredPoints -file $pot_fn -obj /tmp/lsGUI/pot -type ascii

    }
  }

}

proc generate_edge_groups {img path grp edge edgeString} {
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
