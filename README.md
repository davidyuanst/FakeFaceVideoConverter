# FakeFaceVideoConverter
Trained virtual face model, quickly change the face of the video, python tensorflow project for windows 10/11 and no GPU required.

Create a mirror test video:

python videoConverter.py --test ./VideoFile/test.mp4 ./VideoOutput/output1.mp4 cartoonized_pb_train_output_girl_ChineseTangwei_0h

Create a public video:

python videoConverter.py ./VideoFile/test.mp4 ./VideoOutput/output9.mp4 cartoonized_pb_train_output_Man_Japanese3_1h



Install for CPU device:

ffmpeg-python==0.2.0

numpy

opencv-python==4.9.0.80

tensorflow-cpu==2.7.0

tf-slim==1.1.0


Install for GPU(RTX3060 above) device:

ffmpeg-python==0.2.0

numpy

opencv-python==4.9.0.80

tensorflow-gpu==2.7.0

tf-slim==1.1.0
