#!/bin/bash

nvcc -g -G UVMConsistency.cu -o consistency.out -Wno-deprecated-gp
u-targets

for i in $(seq 1 20); do
  ./consistency.out
done

