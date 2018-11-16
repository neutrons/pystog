Example:
./pystog.py --density 0.5 -f ../data/co2_2k_original.sq 0.8 28.0 1 1 0 "S(Q)" -f ../data/ch4_10k_original.sq 0.9 25.0 1 1 0 "S(Q)" --Rmax 25 --Rpoints 2500 --lorch
./pystog.py --density 0.5 -f ../data/co2_2k_original.sq 0.8 28.0 1 1 0 "S(Q)" -f ../data/ch4_10k_original.sq 0.9 25.0 1 1 0 "S(Q)" --Rmax 25 --Rpoints 2500 --fourier-filter-cutoff 1.5 --final-scale 5.0 --lorch-flag
./pystog.py --density 0.5 -f ../data/co2_2k_original.sq 0.8 28.0 1 1 0 "S(Q)" -f ../data/ch4_10k_original.sq 0.9 25.0 1 1 0 "S(Q)" --Rmax 25 --Rpoints 2500 --fourier-filter-cutoff 1.5 --final-scale 5.0 --lorch-flag --mering 0.0 1.0

or using JSON

./pystog.py --json input.json

[![Build Status](https://travis-ci.org/marshallmcdonnell/pystog.svg?branch=master)](https://travis-ci.org/marshallmcdonnell/pystog)
