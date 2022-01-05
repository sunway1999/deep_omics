#!/bin/bash

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 200 -L 1 -e 200 > hs_200.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 100 -L 1 -e 200 > hs_100.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 200 --hidden-size2 100 -L 2 -e 200 > hs_200_100.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 100 --hidden-size2 50 -L 2 -e 200 > hs_100_50.log



sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 200 -L 1 -e 200 --learn-rate 5e-4 > hs_200_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 100 -L 1 -e 200 --learn-rate 5e-4 > hs_100_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 200 --hidden-size2 100 -L 2 -e 200 --learn-rate 5e-4 > hs_200_100_lr_5e-4.log

sleep 0.1; python3 main.py -i mut_matrix_340_x_9112.txt --hidden-size1 100 --hidden-size2 50 -L 2 -e 200 --learn-rate 5e-4 > hs_100_50_lr_5e-4.log
