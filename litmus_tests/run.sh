#!/bin/bash

rm lit.res
rm progress.txt
touch progress.txt
touch lit.res

for i in $(seq 1 100000); do
	if (( i % 100 == 1 )); then
		echo "Iteration $i"	>> progress.txt
	fi
	./a.out >> lit.res
done


