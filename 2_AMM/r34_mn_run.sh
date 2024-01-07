#!/bin/bash

NVALS=(8 1 2 4 8 8 8)
KVALS=(8192 8192 8192 8192 1024 2048 4096)

len=${#NVALS[@]}

for (( i=0; i<$len; i++ ))
do
    n=${NVALS[$i]}
    k=${KVALS[$i]}

    python main.py -g 5 -m r34 -d m -tr 1000 -te 500 -n $n -k $k -s 0,0,0,0,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1 
done

