#!/bin/bash

nvcc -DFIX -g -G UVMConsistency.cu -o consistency.out -Wno-deprecated-gpu-targets

for i in $(seq 1 20); do
  ./consistency.out
done

