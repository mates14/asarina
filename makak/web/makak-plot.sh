#!/bin/bash
# makak-plot.sh NGHT/mrYYYYMMDD.dat: the night's graph, as makak-reloaded's
# bin/makak-plot.sh on zeus drew it from its reprocessing.  Zeropoint, sky
# brightness (BGSIGMA less the camera's dark noise D(T), fitted on the darks of
# mrYYYYMMDDd.dat), CLOUD from D50's meteo log, CCD and ambient temperature.
# Writes mrYYYYMMDD.png and uploads it as makak/makak.png and
# makak/mrpng/mrYYYYMMDD.png, the night's mr*.dat to makak/dat/.
# NOUPLOAD=1 only plots.

base=/home/mates/asarina+makak
cd $base/web || exit 1

file=$1
dfile=${file%.dat}d.dat
ffile=${file%.dat}f.dat
night=$(basename $file .dat); night=${night#mr}
test -e $file && test $(wc -l < $file) -ge 6 || exit 0

now=$(sort -k2 $file | tail -n1 | awk '{printf("%ld\n", ($2-2440587.5)*86400);}')
now2=$(date -ud@$[now+43200] +%Y-%m-%d)
now1a=$(date -ud@$[now-43200] +%Y%m%d)
now2a=$(date -ud@$[now+43200] +%Y%m%d)
midnight=$(date -ud $now2 +%s)

# twilights as seconds from the UT midnight
eval $(rts2-state --UT -N -d $now2 2>/dev/null | awk -v d=86400 '/night/{print "tmnight="$1%d-d;} /dawn/{print "tmdawn="$1%d;} /morning/{print "tmmorning="$1%d;} /dusk/{print "tmdusk="$1%d-d;}')

# CLOUD (sky minus ambient temperature, TEMP_AMB) from D50's rts2-logd
mkdir -p meteo
for d in $now1a $now2a; do
	rsync -q lascaux50.asu.cas.cz:rts2-meteo-$d.log meteo/ 2>/dev/null
done
find meteo -name 'rts2-meteo-*.log' -mtime +3 -delete
cat meteo/rts2-meteo-$now1a.log meteo/rts2-meteo-$now2a.log 2>/dev/null | grep "^CLOUD\>" |
	while read _ a b _ _ _ d; do echo $(date -ud ${a}T$b +%s) $d; done |
	awk "\$1>$midnight+($tmdusk) && \$1<$midnight+$tmmorning" > cloud$$.dat
test -s cloud$$.dat || echo "$midnight NaN" > cloud$$.dat

gnuplot <<END
set grid
h=3600.
d=86400.0

# dark noise of the camera (in ADU, fitted on darks):
g=2.00 # kinda true for g1-2000
A=460.0
B=1.2
N=60.0
D(temp) = 1/g*sqrt(g*A*B**temp+N*N*g*g)
set fit quiet logfile '/dev/null'
fit D(x) "$dfile" u 3:(\$2-D(\$3)<12.5?\$2:NaN) via A

# sky mag/arcsec2 from BGSIGMA and CCD temperature, 1 px = 3212.7 arcsec2
Z=19.4
sky(s,T)=Z-2.5*log10((s*s*g*g-(g*D(T))**2)/g/3212.7)

ts(x)=((x+d/2)/d-int((x+d/2)/d)-0.5)*24
t(x)=(x-int(x)-0.5)*24

set terminal png size 1280,800 large font helvetica 24
set output "mr$night.png"

dz=0.15
Nplot=4
set multiplot

set origin 0,1.0/Nplot
set size 1.05,3.0/Nplot
set xrange [$tmnight/h-5.0/60:$tmdawn/h+5.0/60]
set xtics ("14"-10, "16" -8, "18" -6, "20" -4, "22" -2, 0, 2, 4, 6, 8, 10)
set x2tics ( sprintf("%.0f:%02.0f",24+int($tmnight/h)-1,60+($tmnight-(h*int($tmnight/h)))/60) $tmnight/h, sprintf("%.0f:%02.0f",int($tmdawn/h),($tmdawn-(h*int($tmdawn/h)))/60) $tmdawn/h )
set ytics nomirror
set yrange [19.5:13.5]
set y2tics
set y2range [26.5:-1.5]
set label "$night" center at graph 0.5,0.9 font 'arial,32'

plot "$file" u (t(\$2)):(\$9<dz?\$8:NaN) pt 7 ps 0.7 lt rgb "#dd3333" t "Zeropoint",\
 "$file" u (t(\$2)):(sky(\$14,\$15)) pt 7 ps 0.7 lt rgb "#3333dd" t "sky (mag/arcsec2)",\
 "< cat $ffile 2>/dev/null" u (ts(\$1)):(sky(\$2,\$3)) pt 7 ps 0.7 lt rgb "#3333dd" t "",\
 "cloud$$.dat" u (ts(\$1)):2 pt 7 ps 0.7 lt rgb "#33aa33" t "cloud" axes x1y2

unset label
unset x2tics
unset y2tics
set origin 0,0
set size 2.2/Nplot,1.2/Nplot
set ytics 5
set autoscale y
plot "$file" u (t(\$2)):15 pt 7 ps 0.7 lt 1 t "T_C_C_D", \\
	"$dfile" u (ts(\$1)):3 pt 7 ps 0.7 lt 2 t "", \\
	"cloud$$.dat" u (ts(\$1)):37 pt 7 ps 0.7 lt 4 t "T_A_M_B"

set origin 2.0/Nplot,0
set size 1.9/Nplot,1.2/Nplot
set autoscale x
set xtics 1
set ytics 10
plot "$dfile" u 3:(\$2<150?\$2:NaN) pt 7 t "sigma(T)", D(x)
END
rm -f cloud$$.dat

if [ -z "$NOUPLOAD" ]; then
	rsync mr$night.png lascaux.asu.cas.cz:mrpng/
	rsync mr$night.png lascaux.asu.cas.cz:makak.png
	rsync $file lascaux.asu.cas.cz:dat/
fi
