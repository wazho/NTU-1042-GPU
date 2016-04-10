# Move to the project root.
BASEDIR=$(dirname "$0")
cd $BASEDIR

# Create the debug folder.
rm -rf ./debug
mkdir -p ./debug

# Compiler the CUDA program.
nvcc -c --std=c++11 -arch=sm_30 lab2.cu -o ./debug/lab2.o
nvcc --std=c++11 -arch=sm_30 ./debug/lab2.o main.cu -o ./debug/lab2

# Generate and convert the video format.
cd ./debug
./lab2
avconv -i ./result.y4m video.mp4

# Async the file to another computer.
# rsync or scp

## If wanna trim video and convert to y4m.
## ffmpeg -i volleyball.mp4 -ss 00:02:20 -t 00:00:06 -vf fps=20 -an -c:v rawvideo -f yuv4mpegpipe -y output.y4m
## avconv -i ./output.y4m output.mkv