#!/bin/bash

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 -L 1 -e 10 > hs_400_e_10.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 -L 1 -e 50 > hs_400_e_50.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 600 -L 1 -e 10 > hs_600_e_10.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 600 -L 1 -e 50 > hs_600_e_50.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 --hidden-size2 200 -L 2 -e 10 > hs_400_200_e_10.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 --hidden-size2 200 -L 2 -e 50 > hs_400_200_e_50.log




sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 -L 1 -e 10 --learn-rate 5e-4 > hs_400_e_10_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 -L 1 -e 50 --learn-rate 5e-4 > hs_400_e_50_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 600 -L 1 -e 10 --learn-rate 5e-4 > hs_600_e_10_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 600 -L 1 -e 50 --learn-rate 5e-4 > hs_600_e_50_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 --hidden-size2 200 -L 2 -e 10 --learn-rate 5e-4 > hs_400_200_e_10_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 400 --hidden-size2 200 -L 2 -e 50 --learn-rate 5e-4 > hs_400_200_e_50_lr_5e-4.log
