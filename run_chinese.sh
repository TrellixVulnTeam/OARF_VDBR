#!/bin/bash

# Required files:
# data/HWDB1.1fullset.hdf5, data/HIT_OR3Cfullset.hdf5, data/HIT_HWDB1.1_fullset.hdf5

# experiment 1
python main.py --setting=casia --task=chinese
python main.py --setting=hit --task=chinese
python main.py --setting=combined --task=chinese
python main.py --setting=fedavg --task=chinese  # default is fedavg

# experiment 2
python main.py --local-epochs=1 --task=chinese
python main.py --local-epochs=2 --task=chinese
python main.py --local-epochs=3 --task=chinese
python main.py --local-epochs=4 --task=chinese
python main.py --local-epochs=5 --task=chinese
python main.py --local-epochs=6 --task=chinese
python main.py --local-epochs=7 --task=chinese
python main.py --local-epochs=8 --task=chinese

# experiment 3
python main.py --dp -e=0.03125 --task=chinese
python main.py --dp -e=0.0625 --task=chinese
python main.py --dp -e=0.125 --task=chinese
python main.py --dp -e=0.25 --task=chinese
python main.py --dp -e=0.5 --task=chinese
python main.py --dp -e=1.0 --task=chinese
python main.py --dp -e=2.0 --task=chinese
python main.py --dp -e=4.0 --task=chinese
python main.py --dp -e=6.4 --task=chinese

# experiment 3.1
python main.py --dp -e=2 --lotsize-scaler=0.1 --task=chinese
python main.py --dp -e=2 --lotsize-scaler=1 --task=chinese
python main.py --dp -e=2 --lotsize-scaler=3.1623 --task=chinese
python main.py --dp -e=2 --lotsize-scaler=10 --task=chinese
python main.py --dp -e=2 --lotsize-scaler=31.623 --task=chinese
python main.py --dp -e=2 --lotsize-scaler=100 --task=chinese
python main.py --dp -e=2 --lotsize-scaler=316.23 --task=chinese

python main.py --dp -e=2 --lotsize-scaler=0.1 --setting=casia --task=chinese
python main.py --dp -e=2 --lotsize-scaler=1 --setting=casia --task=chinese
python main.py --dp -e=2 --lotsize-scaler=3.1623 --setting=casia --task=chinese
python main.py --dp -e=2 --lotsize-scaler=10 --setting=casia --task=chinese
python main.py --dp -e=2 --lotsize-scaler=31.623 --setting=casia --task=chinese
python main.py --dp -e=2 --lotsize-scaler=100 --setting=casia --task=chinese
python main.py --dp -e=2 --lotsize-scaler=316.23 --setting=casia --task=chinese

python main.py --dp -e=2 --lotsize-scaler=0.1 --setting=hit --task=chinese
python main.py --dp -e=2 --lotsize-scaler=1 --setting=hit --task=chinese
python main.py --dp -e=2 --lotsize-scaler=3.1623 --setting=hit --task=chinese
python main.py --dp -e=2 --lotsize-scaler=10 --setting=hit --task=chinese
python main.py --dp -e=2 --lotsize-scaler=31.623 --setting=hit --task=chinese
python main.py --dp -e=2 --lotsize-scaler=100 --setting=hit --task=chinese
python main.py --dp -e=2 --lotsize-scaler=316.23 --setting=hit --task=chinese
