#!/bin/bash

nvcc -g -G UVMConsistency.cu -o consistency.out

for i in $(seq 1 20); do
  ./consistency.out
done

