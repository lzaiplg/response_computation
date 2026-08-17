
# 引入模型文件
source 0_material.tcl
source 1_node.tcl
source 2_Section.tcl
source 3_fiber.tcl
source 4_Element.tcl
source 5_Fix.tcl
source 6_mass_load.tcl

# 设置分析
set numModes 10                     
set lambda [eigen $numModes]        

# 输出频率
set omega {}
set f {}
for {set i 0} {$i < $numModes} {incr i} {
    lappend omega [expr sqrt([lindex $lambda $i])]
    lappend f [expr [lindex $omega $i] / (2*3.1415926535)]
}
puts "Frequencies (Hz): $f"
