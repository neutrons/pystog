#!/usr/bin/env bash

bcoh=3.644
num_density=0.02138

md_results="argon_rdf2neutron.dat"
GKofR_col=1
RofR_col=2
rhoR_col=3
GofR_col=4
gofr_col=5
DofR_col=6
TofR_col=7

GKofR="argon.GKofR"
RofR="argon.RofR"
rhoR="argon.rhoR"
GofR="argon.GofR"
gofr="argon.gofr"
DofR="argon.DofR"
TofR="argon.TofR"

sq="argon.sq"
fq="argon.fq"
fq_keen="argon.fq_keen"

xlo=0.01
xhi=59.99
binsize=0.02

# Keen's G(r)
./transformer.py GK_to_FK -i ${md_results} -o ${fq_keen} -b ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col} 
./transformer.py FK_to_GK -i ${fq_keen}    -o ${GKofR}   -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col} -f ${GKofR} 0 1 --title "Keen G(r) -> Keen F(Q) -> Keen G(r)"

./transformer.py GK_to_F -i ${md_results} -o ${fq}    -b ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col}
./transformer.py F_to_GK -i ${fq}         -o ${GKofR} -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col} -f ${GKofR} 0 1 --title "Keen G(r) -> F(Q) -> Keen G(r)"

./transformer.py GK_to_S -i ${md_results} -o ${sq}    -b ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col}
./transformer.py S_to_GK -i ${sq}         -o ${GKofR} -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col}  -f ${GKofR} 0 1 --title "Keen G(r) -> S(Q) -> Keen G(r)"

# PDF = G(r)
./transformer.py G_to_FK -i ${md_results} -o ${fq_keen} -b ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py FK_to_G -i ${fq_keen}    -o ${GofR}    -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col} -f ${GofR} 0 1 --title "G(r) -> Keen F(Q) -> G(r)"

./transformer.py G_to_F -i ${md_results} -o ${fq}   -b ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py F_to_G -i ${fq}         -o ${GofR} -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col} -f ${GofR} 0 1 --title "G(r) -> F(Q) -> G(r)"

./transformer.py G_to_S -i ${md_results} -o ${sq}   -b ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py S_to_G -i ${sq}         -o ${GofR} -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col}  -f ${GofR} 0 1 --title "G(r) -> S(Q) -> G(r)"

# g(r) 
./transformer.py g_to_FK -i ${md_results} -o ${fq_keen} -b ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col} 
./transformer.py FK_to_g -i ${fq_keen}    -o ${gofr}    -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col} -f ${gofr} 0 1 --title "g(r) -> Keen F(Q) -> g(r)"

./transformer.py g_to_F -i ${md_results} -o ${fq}   -b ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col}
./transformer.py F_to_g -i ${fq}         -o ${gofr} -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col} -f ${gofr} 0 1 --title "g(r) -> F(Q) -> g(r)"

./transformer.py g_to_S -i ${md_results} -o ${sq}   -b ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col}
./transformer.py S_to_g -i ${sq}         -o ${gofr} -b ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col}  -f ${gofr} 0  1 --title "g(r) -> S(Q) -> g(r)"

