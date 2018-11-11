#!/bin/bash

file="tmp.txt"
rm $file
touch $file

for i in $(seq 1 2000); do
	if (( i % 10 == 0)); then
		echo "Iteration number $i"
	fi
	./a.out >> $file
done
echo "No Inconsistency Found"
