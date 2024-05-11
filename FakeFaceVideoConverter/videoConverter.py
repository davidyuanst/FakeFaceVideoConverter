
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

import argparse
import ffmpeg
import logging
import numpy as np
import os
import subprocess
import cv2

parser = argparse.ArgumentParser(description='Example streaming ffmpeg numpy processing')
parser.add_argument('in_filename', default = './VideoFile/C141.mp4', type = str, help='Input filename')
parser.add_argument('out_filename', default = './VideoOutput/C141.mp4', type = str, help='Output filename')
parser.add_argument('model_filename', default = 'cartoonized_pb_train_output_girl_Chinese5_4h', type = str, help='Model filename')
parser.add_argument('--test', action='store_true', help='Output comparison video')

DEFAULT_HEIGHT=1280

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def get_new_size(w,h):
    
        height = DEFAULT_HEIGHT

        HWrate=h/w

        width = int(height / HWrate /16.0) * 16
        
        return width,height
        
def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame



def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def run(in_filename, out_filename, process_frame,test):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    newWidth,newHeight=get_new_size(width,height)
    #print("New Size:", newWidth, newHeight)
    if(test):
        if(newWidth>newHeight):
            process2 = start_ffmpeg_process2(out_filename, newWidth, newHeight*2)
        else:
            process2 = start_ffmpeg_process2(out_filename, newWidth*2, newHeight)
    else:
        process2 = start_ffmpeg_process2(out_filename, newWidth, newHeight)
        
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        out_frame = process_frame(in_frame,test)
        write_frame(process2, out_frame)

    logger.info('Waiting for ffmpeg process1')
    process1.wait()

    logger.info('Waiting for ffmpeg process2')
    process2.stdin.close()
    process2.wait()

    logger.info('Done')


class DeepDream(object):
    
    def __init__(self,model_folder):
        pb_path = './output_pb/'+model_folder+'/frozen_model.pb'
        print("DeepDream Init:",pb_path)
        self._sess = tf.Session()
        with gfile.FastGFile(pb_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self._sess.graph.as_default()
            tf.import_graph_def(graph_def,name='')
            #for i,n in enumerate(graph_def.node):
            #    print("Name of the node -%s"%n.name)


        self._sess.run(tf.global_variables_initializer())

        self._input_x = self._sess.graph.get_tensor_by_name('x:0')
        prediction = self._sess.graph.get_tensor_by_name('prediction:0')

        self._pred=tf.transpose(prediction,perm=[0,2,3,1])

        print(self._input_x,prediction,self._pred)

    def process_frame(self, frame,test):
        
        h,w,c=frame.shape
        

        width,height=get_new_size(w,h)
            
        #logger.info('process frame')
        #print("process_frame",width,height)
        
        frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        
        frame_resize8 = cv2.resize(frameBGR, (width, height))
        
        frame_resize = frame_resize8.astype(np.float32)
        frame_resize = frame_resize.reshape([1,height,width,3])

        retImg = self._sess.run(self._pred, {self._input_x:frame_resize})
        
        """
        if (w>h):
            left = int ((width - height) /2)
            right = int(left + height)
            frame_resize_crop = frame_resize[:,:,left:right,:]
            retImgCrop = self._sess.run(self._pred, {self._input_x:frame_resize_crop})
            retImg[:,:,left:right,:] = retImgCrop
        """
        #retImg[:,:,0:int(width/2),:] = frame_resize[:,:,0:int(width/2),:]
        
    
        retImg8 = retImg.astype(np.uint8).reshape([height,width,3])
        if(test):
            if(h>w):
                retImg2 = np.concatenate((frame_resize8,retImg8),axis=1)
            else:
                retImg2 = np.concatenate((frame_resize8,retImg8),axis=0)
        else:
            retImg2 = retImg8
        #rerFrame = cv2.resize(retImg.astype(np.float32), (w, h))
        rerFrameRGB = cv2.cvtColor(retImg2, cv2.COLOR_BGR2RGB) 
        
        return rerFrameRGB


if __name__ == '__main__':
    args = parser.parse_args()
    process_frame = DeepDream(args.model_filename).process_frame
    run(args.in_filename, args.out_filename,process_frame,args.test)
