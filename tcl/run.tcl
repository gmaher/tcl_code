set where [file dirname [info script]]
source [file join $where lsVTKgen.tcl]

runGroupsToVTKonFiles images.txt paths.txt groups.txt edge.txt
