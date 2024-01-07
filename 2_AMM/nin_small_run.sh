#!/bin/bash

NVALS=(1 2 4 8 8 8 8)
KVALS=(8192 8192 8192 1024 2048 4096 8192)

len=${#NVALS[@]}

for (( i=0; i<$len; i++ ))
do
    n=${NVALS[$i]}
    k=${KVALS[$i]}

    python main.py -g 3 -m n -d c100 -tr 1000 -te 500 -n $n -k $k -s 1,0,0,1,1,0,0,1,1 

done

