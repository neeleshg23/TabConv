#!/bin/bash
N=4
K=(4096 8192 16384 32768)
S="1,1,0,0,1,0" 

for k in "${K[@]}"
do
    nohup python main.py -m r18 -d c10 -n $N -k $k -s $S -tr 1000 -te 500 -g 5 &
done
