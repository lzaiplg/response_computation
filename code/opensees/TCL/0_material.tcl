set m 1.
set kg 1.
set sec 1.

set LunitTXT "m"
set FunitTXT "N"
set TunitTXT "sec"

set dm [expr 0.1*$m]
set cm [expr 0.01*$m]
set mm [expr 0.001*$m]

set N  [expr $kg*$m/pow($sec,2)]
set kN [expr 1000.*$N]

set g [expr 9.8*$m/pow($sec,2)]

set m2 [expr $m*$m]
set m4 [expr $m*$m*$m*$m]
set mm2 [expr $mm*$mm]
set mm4 [expr $mm*$mm*$mm*$mm]

set Pa  [expr $N/pow($m,2)]
set kPa [expr 1000.*$Pa]
set MPa [expr 1000.*$kPa]
set GPa [expr 1000.*$MPa]
set PI [expr acos(-1.0)]

set sec 1.0
