set term postscript 16
set output 'sources.ps'
set size ratio -1
set xrange [-0:100]
set yrange [-0:100]
plot 'sources.dat' u 3:2 notitle w p ,'stations-GPS.dat' u 2:1:4 notitle w labels,'stations-GPS.dat' u 2:1:3 notitle w p, 'GPSarrowsD.dat' u 2:1:4:3 w vectors, 'GPSarrows.dat' u 2:1:4:3 w vectors
