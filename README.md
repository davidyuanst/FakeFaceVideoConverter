# Fake Face Video Converter

The purpose of the system is mainly for network anchors whose looks are not perfect enough, so that they can apply makeup to optimize their faces/hair without having to hide behind the screen and only provide voice.

Pretrained fake face model, quickly change the face of the video, python tensorflow project for windows 10/11 and no GPU required.

All pre-trained models use the StableDiffusion random algorithm to generate virtual characters, which are not taken from real people's faces, nor do they correspond to someone in the real world.
In order to ensure privacy and prevent abuse, training code and technology are not provided, and only pre-trained models can be used.

<b>Features:</b>

Highest resolution: (unlimited width) x (800-1680)

Speed: 1fps (i5 CPU) 15fps (RTX3060)

<b>Models folder:</b> ./FakeFaceVideoConverter/model_pb/

<b>Create test video:</b> python videoConverter.py --test ./VideoFile/9_04_09_19.mp4 ./VideoOutput/output1.mp4 ./model_pb/pb_train_output_girl_ChineseTangwei_0h

<b>Create release video:</b> python videoConverter.py ./VideoFile/4216631.mp4 ./VideoOutput/output12.mp4 ./model_pb/pb_train_output_Man_Japanese3_1h


<b>Example output Videos:</b>
[FakeFaceVideoConverter/VideoOutput](https://github.com/davidyuanst/FakeFaceVideoConverter/tree/main/FakeFaceVideoConverter/VideoOutput)

[![Example Video](https://img.youtube.com/vi/hTNy9AyH82s/maxresdefault.jpg)](https://www.youtube.com/watch?v=hTNy9AyH82s)
Youtube Video Example


<b>Example input Videos:</b>
[https://www.pexels.com/search/videos/girl/](https://www.pexels.com/search/videos/girl/)


<b>Install to CPU device:</b>

Python==3.7.12

ffmpeg [download](https://ffmpeg.org/download.html)

ffmpeg-python==0.2.0

mtcnn                        0.1.1

numpy

opencv-python==4.9.0.80

tensorflow-cpu==2.7.0

tf-slim==1.1.0


<b>Install to GPU(RTX3060 above) device:</b>

Python==3.7.12

ffmpeg [download](https://ffmpeg.org/download.html)

ffmpeg-python==0.2.0

mtcnn                        0.1.1

numpy

opencv-python==4.9.0.80

tensorflow-gpu==2.7.0

tf-slim==1.1.0
