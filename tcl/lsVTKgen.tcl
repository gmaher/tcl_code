proc runGroupsToVTKonFiles {imgs paths groups edges} {
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
	set edges [readFromFile $edges]

  global itklsGUIParams
	set itklsGUIParams(phyRadius) 0.3

	foreach I $imgs P $paths G $groups E $edges {
		puts "$I\n $P\n $G\n"
		catch {generate_truth_groups $I $P $G}
		catch {generate_edge_groups $I $P $G}
		catch {generate_edge_groups $I $P $G $E "edge96"}
		close_files


		#generate_edge_groups $I $P $G $E "edge96"
		#generate_edge_groups $I $P $G $E "edge96_LS" LSEdge
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

	#reset edgeImage param
	global itklsGUIParams
	set itklsGUIParams(edgeImage) 0
	set itklsGUIParams(2DEdgeImage) "image"

	#file for storing group info
	global pathInfoFile
	global gPathPoints

  puts "starting group loop"
  foreach grp [array names pathmap] {
		puts $grp
		#if {[string match "LVA" $grp]} {
		#	set data [gets stdin]
		#}
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

				puts $pathInfoFile "${imgname} ${grp} ${point} [lindex $gPathPoints($pathid,splinePts) $point]"

				if {[repos_exists -obj /lsGUI/$pathid/$point/thr/selected]} {
					puts "ls 1"
	      	repos_writeVtkPolyData -file $ls_fn -obj /lsGUI/$pathid/$point/thr/selected -type ascii
	      	repos_writeVtkStructuredPoints -file $mag_fn -obj /tmp/lsGUI/mag -type ascii
	      	repos_writeVtkStructuredPoints -file $pot_fn -obj /tmp/lsGUI/pot -type ascii

					#Delete repository objects so they don't hang around
					repos_delete -obj /lsGUI/$pathid/$point/thr/selected
					repos_delete -obj /tmp/lsGUI/mag
					repos_delete -obj /tmp/lsGUI/pot
				}
				if {[repos_exists -obj /lsGUI/$pathid/$point/ls]} {
					puts "ls 2"
					repos_writeVtkPolyData -file $ls_fn -obj /lsGUI/$pathid/$point/ls -type ascii
					repos_writeVtkStructuredPoints -file $mag_fn -obj /tmp/lsGUI/mag -type ascii
					repos_writeVtkStructuredPoints -file $pot_fn -obj /tmp/lsGUI/pot -type ascii

					repos_delete -obj /lsGUI/$pathid/$point/ls
					repos_delete -obj /tmp/lsGUI/mag
					repos_delete -obj /tmp/lsGUI/pot
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
	puts "generated edge"
  puts "generate_edge_groups ${img} ${path} ${grp} ${edge}"
  global gImageVol
  global gFilenames
  load_files $img $path $grp $edge

  global itklsGUIParams
	set itklsGUIParams(phyRadius) 0.3

  if {$edge ne 0} {
		set itklsGUIParams(2DEdgeImage) $edgeType
		set itklsGUIParams(useEdgeImage) "disp"
	} else {
		set itklsGUIParams(2DEdgeImage) "image"
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
		if {[group_exists $grp]} {
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

				repos_delete -obj /lsGUI/$pathid/$point/ls
				repos_delete -obj /img/$pathid/$point/mag
				repos_delete -obj /img/$pathid/$point/pot
	    }
		}
  }
	puts "closing files"
	close_files
}

proc get_image_name {fn} {
  #function to parse the filepath to an image to get only the image name
  #
  #args:
  #   @a fn - string, the filepath to the image
  set f [lindex [split $fn /] end]
	string map {_seg ""} $f
  string map {-cm.mha ""} $f
	string map {_all.mha ""} $f
	string map {-image.mha ""} $f
	return $f
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
	puts "loading ${img} ${path} ${grp} ${edge}"
  global gImageVol

  if {[string match *.mha* $img]} {
    set gImageVol(mha_filename) $img
  } else {
    set gImageVol(xml_filename) $img
  }
	puts "loading edge"
  if {$edge ne "0"} {
    seg_LoadEdgeImageMha $edge
  }
  createPREOPloadsaveLoadVol

  #load paths
	puts "loading path"
	puts $path
  global gFilenames
  set gFilenames(path_file) $path
  guiFNMloadHandPaths

  #load groups
	puts "loading surfaces"
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

proc checkGroupsInPathPoints {grp} {
	global gPathPoints
	set groupmembers [group_get $grp]
	foreach guy $groupmembers {
		puts $guy
		if [catch {set pathId [repos_getLabel -obj $guy -key paste_pathId]}] {
				set pathId [repos_getLabel -obj $guy -key pathId]
			if {[info exists gPathPoints($pathId,numSplinePts)]}{
				puts "exists"
			}
			else {
				return 0
			}
		}
		else {
			return 0
		}
	}
	return 1
}

proc group_names {} {
    #@author Ken Wang
    #@c Return the group names.
    #@r list of group names.
    global gGroup
    return [array names gGroup]
}

proc model_loop {folder} {
	set group_folders [glob -type d $folder*]

	global fn
	set fn [open "log.txt" "w"]
	close $fn
	foreach fold $group_folders {
		catch {
			model_groups $fold
		}
	}
}

proc model_groups {fold} {
	global guiBOOLEANvars
	global gFilenames

	puts "loading surfaces $fold"
	if {[llength [glob $fold/*]] <= 1} {
		puts "empty groups folder continuing"
		return 0
	}

	set gFilenames(groups_dir) $fold
	createPREOPgrpLoadGroups
	guiSV_group_update_tree

	set groups [group_names]
	set img [get_image_name $fold]
	global fn

	foreach grp $groups {
    	set numSegs [llength [group_get $grp]]
			if {$numSegs > 1} {
				set fn [open "log.txt" "a"]
				puts $img.$grp
				puts $fn $img.$grp
				close $fn
				set guiBOOLEANvars(selected_groups) $grp
				create_model_polydata $grp
				repos_writeVtkPolyData -file $img.$grp.vtp -obj /models/PolyData/$grp -type ascii
			}

	}

	close_files
}

proc create_model_polydata {model_name} {
  global gRen3d
  global guiPDvars
  global gui3Dvars
  global gOptions
  global guiSVvars

  global guiBOOLEANvars
  set gOptions(meshing_solid_kernel) PolyData
  set kernel $gOptions(meshing_solid_kernel)
  set gOptions(meshing_solid_kernel) $kernel
  solid_setKernel -name $kernel

  set ordered_names    $guiBOOLEANvars(selected_groups)
  set ordered_names2    $guiBOOLEANvars(selected_seg3d)
  set sampling_default $guiBOOLEANvars(sampling_default)
  set lin_multiplier   $guiBOOLEANvars(linear_sampling_along_length_multiplier)
  set useLinearSampleAlongLength   $guiBOOLEANvars(use_linear_sampling_along_length)
  set numModes         $guiBOOLEANvars(num_modes_for_FFT)
  array set overrides  $guiBOOLEANvars(sampling_overrides)
  set useFFT           $guiBOOLEANvars(use_FFT)
  set sample_per_segment $guiBOOLEANvars(sampling_along_length_multiplier)
  set addCaps          $guiBOOLEANvars(add_caps_to_vessels)
  set noInterOut       $guiBOOLEANvars(no_inter_output)
  set tol 	       $guiBOOLEANvars(tolerance)
  set spline           $guiBOOLEANvars(spline_type)


  #set model [guiSV_model_new_surface_name 0]
	set model $model_name
  catch {repos_delete -obj $model}
  if {[model_create $kernel $model] != 1} {
    guiSV_model_delete_model $kernel $model
    catch {repos_delete -obj /models/$kernel/$model}
    model_create $kernel $model
  }

  foreach grp $ordered_names {

    set numOutPtsInSegs $sampling_default

    if [info exists overrides($grp)] {
      set numOutPtsInSegs $overrides($grp)
      puts "overriding default ($sampling_default) with ($numOutPtsInSegs) for ($grp)."
    }

    set vecFlag 0
		global fn

    set numSegs [llength [group_get $grp]]

    set numOutPtsAlongLength [expr $sample_per_segment * $numSegs]

    set numPtsInLinearSampleAlongLength [expr $lin_multiplier *$numOutPtsAlongLength]

		set fn [open "log.txt" "a"]
		puts $fn $numSegs.$numOutPtsAlongLength.$numPtsInLinearSampleAlongLength
		close $fn

    puts "num pts along length: $numPtsInLinearSampleAlongLength"

    set outPD /guiGROUPS/polydatasurface/$grp
    catch {repos_delete -obj $outPD}

    solid_setKernel -name PolyData
    polysolid_c_create_vessel_from_group $grp $vecFlag  $useLinearSampleAlongLength $numPtsInLinearSampleAlongLength  $useFFT $numModes  $numOutPtsInSegs $numOutPtsAlongLength $addCaps $spline $outPD

  }

  #if {[llength $ordered_names] ==0 && [llength $ordered_names2] == 0} {
  #    tk_messageBox -title "Select Surfaces on Segmentation Tab"  -type ok -message "Please select surfaces for boolean on segmentation tab"
  #    return
  #}
  for {set i 0} {$i < [llength $ordered_names]} {incr i} {
    lappend vessel_names /guiGROUPS/polydatasurface/[lindex $ordered_names $i]
    lappend names [lindex $ordered_names $i]
  }
  global gSeg3D
  for {set i 0} {$i < [llength $ordered_names2]} {incr i} {
    set name [lindex $ordered_names2 $i]
    lappend names $name
    if {![check_surface_for_capids $gSeg3D($name)]} {
      puts "vessel $name doesn't have CapIDs, setting to -1"
      set_capids_for_pd $gSeg3D($name)
    }
    lappend vessel_names $gSeg3D($name)

  }
  puts $vessel_names

  geom_all_union -srclist $vessel_names -intertype $noInterOut -result $model -tolerance $tol
  set pdobject /models/$kernel/$model
  catch {repos_delete -obj $pdobject}
  $model GetPolyData -result $pdobject
  set rtnstr [geom_checksurface -src $pdobject -tolerance $tol]

  set_facenames_as_groupnames $model $names $addCaps
  #set_facenames_as_groupnames $myresult $names 0

  guiSV_model_add_faces_to_tree $kernel $model
  #guiSV_model_display_only_given_model $model 1

	puts "Model creation $model_name complete"
  #tk_messageBox -title "Solid Unions Complete"  -type ok -message "Number of Free Edges: [lindex $rtnstr 0]\n (Free edges are okay for open surfaces)\n Number of Bad Edges: [lindex $rtnstr 1]"
}
