#!/bin/bash

nvcc -DFIX -g -G UVMConsistency.cu -o consistency.out -Wno-deprecated-gpu-targets

rm tmp.txt
touch tmp.txt

for i in $(seq 1 20000); do
	if (( i % 10 == 0 )); then
		echo "Iteration $i"
	fi
  ./consistency.out	>> tmp.txt
done

