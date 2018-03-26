#!/usr/bin/env bash

bcoh=3.644
btot=5.435
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
dcs="argon.dcs"

xlo=0.01
xhi=59.99
binsize=0.02

# Keen's G(r)
./transformer.py GK_to_FK -i ${md_results} -o ${fq_keen} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col} 
./transformer.py FK_to_GK -i ${fq_keen}    -o ${GKofR}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col} -f ${GKofR} 0 1 --title "Keen G(r) -> Keen F(Q) -> Keen G(r)"


./transformer.py GK_to_F -i ${md_results} -o ${fq}    --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col}
./transformer.py F_to_GK -i ${fq}         -o ${GKofR} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col} -f ${GKofR} 0 1 --title "Keen G(r) -> F(Q) -> Keen G(r)"

./transformer.py GK_to_S -i ${md_results} -o ${sq}    --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col}
./transformer.py S_to_GK -i ${sq}         -o ${GKofR} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col}  -f ${GKofR} 0 1 --title "Keen G(r) -> S(Q) -> Keen G(r)"

./transformer.py GK_to_DCS -i ${md_results} -o ${dcs}    --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GKofR_col}
./transformer.py DCS_to_GK -i ${dcs}         -o ${GKofR} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GKofR_col}  -f ${GKofR} 0 1 --title "Keen G(r) -> DCS(Q) -> Keen G(r)"

# PDF = G(r)
./transformer.py G_to_FK -i ${md_results} -o ${fq_keen} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py FK_to_G -i ${fq_keen}    -o ${GofR}    --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col} -f ${GofR} 0 1 --title "G(r) -> Keen F(Q) -> G(r)"

./transformer.py G_to_F -i ${md_results} -o ${fq}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py F_to_G -i ${fq}         -o ${GofR} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col} -f ${GofR} 0 1 --title "G(r) -> F(Q) -> G(r)"

./transformer.py G_to_S -i ${md_results} -o ${sq}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py S_to_G -i ${sq}         -o ${GofR} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col}  -f ${GofR} 0 1 --title "G(r) -> S(Q) -> G(r)"

./transformer.py G_to_DCS -i ${md_results} -o ${dcs}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${GofR_col}
./transformer.py DCS_to_G -i ${dcs}         -o ${GofR} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${GofR_col}  -f ${GofR} 0 1 --title "G(r) -> DCS(Q) -> G(r)"

# g(r) 
./transformer.py g_to_FK -i ${md_results} -o ${fq_keen} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col} 
./transformer.py FK_to_g -i ${fq_keen}    -o ${gofr}    --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col} -f ${gofr} 0 1 --title "g(r) -> Keen F(Q) -> g(r)"

./transformer.py g_to_F -i ${md_results} -o ${fq}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col}
./transformer.py F_to_g -i ${fq}         -o ${gofr} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col} -f ${gofr} 0 1 --title "g(r) -> F(Q) -> g(r)"

./transformer.py g_to_S -i ${md_results} -o ${sq}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col}
./transformer.py S_to_g -i ${sq}         -o ${gofr} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col}  -f ${gofr} 0  1 --title "g(r) -> S(Q) -> g(r)"

./transformer.py g_to_DCS -i ${md_results} -o ${dcs}   --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 3 --ycol ${gofr_col}
./transformer.py DCS_to_g -i ${dcs}         -o ${gofr} --btot-sqrd ${btot} --bcoh-sqrd ${bcoh} --rho ${num_density} -s 2 -x ${xlo} ${xhi} ${binsize}
./xyplot.py -s 3 -f ${md_results} 0 ${gofr_col}  -f ${gofr} 0  1 --title "g(r) -> DCS(Q) -> g(r)"
