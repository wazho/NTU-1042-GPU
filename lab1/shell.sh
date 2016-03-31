mkdir -p ./output
nvcc -c --std=c++11 -arch=sm_30 counting.cu -o ./output/counting.o
nvcc --std=c++11 -arch=sm_30 ./output/counting.o main.cu -o ./output/lab1
./output/lab1