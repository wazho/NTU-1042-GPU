# Move to the project root.
BASEDIR=$(dirname "$0")
cd $BASEDIR

# Create the debug folder.
rm -rf ./debug
mkdir -p ./debug

# Compile the CUDA program.
nvcc -c --std=c++11 -arch=sm_30 lab3.cu -o ./debug/lab3.o
nvcc -c --std=c++11 -arch=sm_30 pgm.cpp -o ./debug/pgm.o
nvcc --std=c++11 -arch=sm_30 ./debug/lab3.o ./debug/pgm.o main.cu -o ./debug/lab3

# Export image.
cd ./debug
./lab3 ../resources/img_background.ppm ../resources/img_target.ppm ../resources/img_mask.pgm 130 600 ./output.ppm
