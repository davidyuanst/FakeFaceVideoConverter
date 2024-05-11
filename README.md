# Fake Face Video Converter
Trained virtual face model, quickly change the face of the video, python tensorflow project for windows 10/11 and no GPU required.

<b>Features:</b>

Highest resolution: (unlimited width) x 1280

Speed: 1/fps (i5 CPU)



<b>Create test video:</b> python videoConverter.py --test ./VideoInput/test.mp4 ./VideoOutput/output1.mp4 cartoonized_pb_train_output_girl_ChineseTangwei_0h

<b>Create release video:</b> python videoConverter.py ./VideoInput/test.mp4 ./VideoOutput/output9.mp4 cartoonized_pb_train_output_Man_Japanese3_1h


<b>Example output Videos:</b>
[https://github.com/davidyuanst/FakeFaceVideoConverter/FakeFaceVideoConverter/VideoOutput](https://github.com/davidyuanst/FakeFaceVideoConverter/tree/main/FakeFaceVideoConverter/VideoOutput)

<b>Example input Videos:</b>
[https://www.pexels.com/search/videos/girl/](https://www.pexels.com/search/videos/girl/)


<b>Install to CPU device:</b>

Python==3.7

ffmpeg

ffmpeg-python==0.2.0

numpy

opencv-python==4.9.0.80

tensorflow-cpu==2.7.0

tf-slim==1.1.0


<b>Install to GPU(RTX3060 above) device:</b>

Python==3.7

ffmpeg

ffmpeg-python==0.2.0

numpy

opencv-python==4.9.0.80

tensorflow-gpu==2.7.0

tf-slim==1.1.0
