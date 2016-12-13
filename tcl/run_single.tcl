set where [file dirname [info script]]
source [file join $where lsVTKgen.tcl]
puts "There are $argc arguments to this script"
puts "The name of this script is $argv0"
if {$argc > 0} {puts "The other arguments are: $argv" }

set I [lindex $argv 1]
set P [lindex $argv 2]
set G [lindex $argv 3]
set E [lindex $argv 4]
generate_truth_groups $I $P $G
generate_edge_groups $I $P $G
generate_edge_groups $I $P $G $E "edge96"
#mainGUIexit 1
exit
