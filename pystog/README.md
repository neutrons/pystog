Example:
./pystog.py 0.5 -f ../data/co2_2k_original.sq 0.8 28.0 1 1 0 -f ../data/ch4_10k_original.sq 0.9 25.0 1 1 0 --Rmax 25 --Rpoints 2500 --lorch
./pystog.py 0.5 -f ../data/co2_2k_original.sq 0.8 28.0 1 1 0 -f ../data/ch4_10k_original.sq 0.9 25.0 1 1 0 --Rmax 25 --Rpoints 2500 --fourier-filter-cutoff 1.5 --final-scale 5.0 --lorch-flag

Debug example:
./pystog.py 1.0 -f ../data/co2_2k_original.sq 0.8 10.0 1 1 0 --Rmax 25 --Rpoints 2500 --fourier-filter-cutoff 1.5 --final-scale 5.0 --lorch-flag
