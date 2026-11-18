
Re-organizing the files:

```

cd /Users/m.wehrens/Data_UVA/2026_11_eating-dispersion-leaves-thrips/DATA/202611_timeseries/cy7_back_cut

mkdir control
mkdir thrips

mkdir control/3H
mkdir control/6H
mkdir control/9H
mkdir control/12H
mkdir thrips/3H
mkdir thrips/6H
mkdir thrips/9H
mkdir thrips/12H 

mv *3H_*control_* control/3H/
mv *6H_*control_* control/6H/
mv *9H_*control_* control/9H/
mv *12H_*control_* control/12H/

mv *3H_*thrips_* thrips/3H/
mv *6H_*thrips_* thrips/6H/
mv *9H_*thrips_* thrips/9H/
mv *12H_*thrips_* thrips/12H/

cd /Users/m.wehrens/Data_UVA/2026_11_eating-dispersion-leaves-thrips/DATA/202611_timeseries/cy7_front_cut

mkdir control
mkdir thrips

mkdir control/3H
mkdir control/6H
mkdir control/9H
mkdir control/12H
mkdir thrips/3H
mkdir thrips/6H
mkdir thrips/9H
mkdir thrips/12H 

mv *3H_*control_* control/3H/
mv *6H_*control_* control/6H/
mv *9H_*control_* control/9H/
mv *12H_*control_* control/12H/

mv *3H_*thrips_* thrips/3H/
mv *6H_*thrips_* thrips/6H/
mv *9H_*thrips_* thrips/9H/
mv *12H_*thrips_* thrips/12H/
```

