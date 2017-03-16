set where [file dirname [info script]]
source [file join $where lsVTKgen.tcl]
puts "There are $argc arguments to this script"
puts "The name of this script is $argv0"
if {$argc > 0} {puts "The other arguments are: $argv" }

set G [lindex $argv 1]


global gOptions
set gOptions(resliceDims) {256 256}



model_groups $G


mainGUIexit 1
#exit
