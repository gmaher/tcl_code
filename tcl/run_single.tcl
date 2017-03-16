set where [file dirname [info script]]
source [file join $where lsVTKgen.tcl]
puts "There are $argc arguments to this script"
puts "The name of this script is $argv0"
if {$argc > 0} {puts "The other arguments are: $argv" }

set I [lindex $argv 1]
set P [lindex $argv 2]
set G [lindex $argv 3]
set E [lindex $argv 4]

global gOptions
set gOptions(resliceDims) {256 256}

global pathInfoFile
set pathInfoFile [open "[pwd]/../pathInfo.txt" a]

catch {generate_truth_groups $I $P $G}
catch {generate_edge_groups $I $P $G}
catch {generate_edge_groups $I $P $G $E "edge96"}

set S [string map {.mha _seg.mha} $I]
if {[file exists $S]} {
  puts "generating segmentation contours $S"
  generate_edge_groups $S $P $G 0 "seg3d"
}
close $pathInfoFile
mainGUIexit 1
#exit
