#!/bin/bash

nvcc -std=c++11 -DFIX -g -G UVMConsistency.cu -o consistency.out -Wno-deprecated-gpu-targets

rm tmp.txt
touch tmp.txt

for i in $(seq 1 2000); do
	if (( i % 10 == 0 )); then
		echo "Iteration $i"
	fi
  ./consistency.out	>> tmp.txt
done

