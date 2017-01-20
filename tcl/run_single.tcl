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
set gOptions(resliceDims) {64 64}

global pathInfoFile
set pathInfoFile [open "[pwd]/../pathInfo.txt" a]

generate_truth_groups $I $P $G
generate_edge_groups $I $P $G
generate_edge_groups $I $P $G $E "edge96"

close $pathInfoFile
#mainGUIexit 1
exit
