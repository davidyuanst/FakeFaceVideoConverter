# FakeFaceVideoConverter
Trained virtual face model, quickly change the face of the video, python tensorflow project for windows 10/11 and no GPU required.

Features:

Highest resolution: unlimited width X 1280

Speed: 1/fps (i9 CPU)



Create a mirror test video:

python videoConverter.py --test ./VideoFile/test.mp4 ./VideoOutput/output1.mp4 cartoonized_pb_train_output_girl_ChineseTangwei_0h

Create a public video:

python videoConverter.py ./VideoFile/test.mp4 ./VideoOutput/output9.mp4 cartoonized_pb_train_output_Man_Japanese3_1h





Install to CPU device:

Python==3.7

ffmpeg

ffmpeg-python==0.2.0

numpy

opencv-python==4.9.0.80

tensorflow-cpu==2.7.0

tf-slim==1.1.0


Install to GPU(RTX3060 above) device:

Python==3.7

ffmpeg

ffmpeg-python==0.2.0

numpy

opencv-python==4.9.0.80

tensorflow-gpu==2.7.0

tf-slim==1.1.0
