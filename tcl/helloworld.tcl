#!/usr/bin/tclsh

#this is a comment
#command arg1 arg2 arg3 ...
puts "Hello World!"

#special variables
puts $argv0

puts $argc

puts $argv

#comand substitution
puts [expr 1 + 6 + 9]

#variables
set a 3
puts $a

#string
set myString {this is a string}
puts $myString

#list
set myList {red 42 blue}
puts [lindex $myList 2]

#associative arrays
set hashmap(name) gabriel
set hashmap(age) 26
puts $hashmap(name)
puts $hashmap(age)

#floating point
set fl1 10.0
set fl2 3.0
puts [expr $fl1/$fl2]

#logical operators
puts [expr 1 || 0]
puts [expr 1 && 0]
puts [expr $a == 3 ? true:false]

#if statements
if {$a == 3} {
	puts "a = 3"
} else {
	puts "a != 3"
}

#for loops
for {set i 0} {$i <6} {incr i} {
	puts "i = $i"
}

#arrays
set arr(0) tcl
set arr(1) 2
set arr(2) "Hey there"

for {set index 0} {$index < [array size arr]} {incr index} {
	puts $arr($index)
}

#string comparison
puts [string compare hello hello]

#append
set s1 "hello"
append s1 " world"
puts $s1

#list length
set var {orange blue red green}
puts [llength $var]
set var [lsort $var]
puts $var

#dictionary
dict set colours colour1 red
puts $colours
dict set colours colour2 blue
puts $colours
puts [dict size $colours]
puts [dict values $colours]

#procedures
# proc procedureName {arg1 arg2 arg3 ...} {
#	body
#}

proc add {a b} {
	return [expr $a + $b]
}
puts [add 10 30]

proc avg {numbers} {
	set sum 0
	foreach number $numbers {
		set sum [expr $sum + $number]
	}
	set average [expr $sum/[llength $numbers]]
	return $average
}
puts [avg {70 80 50 60}]