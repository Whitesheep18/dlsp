#!/bin/bash

for i in {1..10}
do
   python synthetic_train.py --architecture cnn --initialization default --epochs 3 --tags init
   python synthetic_train.py --architecture cnn --initialization FIR --epochs 3 --tags init
   python synthetic_train.py --architecture cnn --initialization FIR+He --epochs 3 --tags init
   python synthetic_train.py --architecture cnn --initialization He --epochs 3 --tags init
done
