# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import sys
import matplotlib.pyplot as plt

sys.path.append(r'D:\users\changmingliu\models\research')
sys.path.append(r'D:\users\changmingliu\models\research\slim')

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
from research.object_detection.utils import dataset_util
# from research.object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('format','imagelabel','label format.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS
#2560
fixsize_h=256
fixsize_w=320
scale=0.5
blocksize=64
maxval=255
# threshold=maxval*scale*scale
threshold=0
def process_label(labelimage1):
    # labelimage=np.array(labelimage)
    ind = labelimage1 > 128
    ind1 = labelimage1 <= 128
    labelimage=labelimage1
    labelimage[ind] = 255
    labelimage[ind1] = 0
    for i in range(int(fixsize_w/blocksize)):
        for j in range(int(fixsize_h/blocksize)):
            slice=labelimage[i*blocksize:(i+1)*blocksize,j*blocksize:(j+1)*blocksize]
            if np.mean(slice)>threshold:
                labelimage[i * blocksize:(i + 1) * blocksize, j * blocksize:(j + 1) * blocksize]=255
            else:
                labelimage[i * blocksize:(i + 1) * blocksize, j * blocksize:(j + 1) * blocksize]=0
    # return PIL.Image.fromarray(labelimage)
    label=np.zeros((int(fixsize_w/blocksize),int(fixsize_h/blocksize)))
    for x in range(int(fixsize_w/blocksize)):
        for y in range(int(fixsize_h/blocksize)):
            label[x][y]=labelimage[x*blocksize+int(blocksize/2)][y*blocksize+int(blocksize/2)]
    return PIL.Image.fromarray(np.transpose(label.astype(np.uint8)))


def dict_to_tf_example(example,
                       labelexample,
                       image_subdirectory,
                       label_subdirectory,
                       idx,
                       label_map_dict=None):
    """Convert label and file to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      example: image filename
      data: a parsed label file which is a list
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    """
    img_path = os.path.join(image_subdirectory, example)
    # with tf.gfile.GFile(img_path, 'rb') as fid:
    #     encoded_jpg = fid.read()
    label_path=os.path.join(label_subdirectory,labelexample)
    # with tf.gfile.GFile(label_path, 'rb') as fid:
    #     encoded_label = fid.read()
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    # image = PIL.Image.open(encoded_jpg_io)
    # encoded_label_io = io.BytesIO(encoded_label)
    # labelimg = PIL.Image.open(encoded_label_io)
    image = PIL.Image.open(img_path)
    w,h=image.size
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    image=image.resize((fixsize_w,fixsize_h),PIL.Image.BILINEAR)
    saveimgpath=os.path.join(os.path.join(image_subdirectory,'..'),'resizedimages')


    if FLAGS.format=="imagelabel":
        labelimage=PIL.Image.open(label_path)
        labelimage = labelimage.resize((fixsize_w, fixsize_h), PIL.Image.BILINEAR)
        # labelimage=np.array(labelimage)
        labelimage = np.transpose(labelimage)
        labelimage.flags.writeable=True
    elif FLAGS.format=="txtlabel":
        data = parse_label(label_path)
        labelimage = np.zeros((fixsize_w, fixsize_h), dtype=int)
        labelimage = setlabel(labelimage, data,w,h)
        # plt.imshow(labelimage)
        labelimage=np.array(labelimage)
        # plt.imshow(labelimage1)
        
        
    
    labelimage=process_label(labelimage)

    savelabelpath = os.path.join(os.path.join(label_subdirectory,'..'), 'resizedCTUlabels')
    # import shutil
    # if os.path.exists(saveimgpath):
    #     shutil.rmtree(saveimgpath)
    # if os.path.exists(savelabelpath):
    #     shutil.rmtree(savelabelpath)
    if not os.path.exists(saveimgpath):
        os.makedirs(saveimgpath)
    if not os.path.exists(savelabelpath):
        os.makedirs(savelabelpath)
    # path_i=os.path.join(saveimgpath, example)
    path_i = os.path.join(saveimgpath, 'img_'+str(idx).zfill(5)+'.jpg')
    path_l=os.path.join(savelabelpath,os.path.splitext(labelexample)[0]+'.png')
    image.save(path_i)
    labelimage.save(path_l)
    width, height = image.size

    # with tf.gfile.GFile(path_i, 'rb') as fid:
    #    encoded_jpg = fid.read()
    # with tf.gfile.GFile(path_l, 'rb') as fid:
    #     encoded_label = fid.read()
    #     # encoded_label = io.BytesIO(encoded_label)
    #     # label = PIL.Image.open(encoded_label)
    #     # label_=np.array(label)
    #     # import matplotlib.pyplot as plt
    #     # plt.imshow(label)
    # key = hashlib.sha256(encoded_jpg).hexdigest()
    # feature_dict = {
    #     'image/height': dataset_util.int64_feature(height),
    #     'image/width': dataset_util.int64_feature(width),
    #     'image/filename': dataset_util.bytes_feature(
    #         example.encode('utf8')),
    #     'image/source_id': dataset_util.bytes_feature(
    #         example.encode('utf8')),
    #     'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    #     'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    #     'image/label/encoded':dataset_util.bytes_feature(encoded_label),
    # }
    # example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    # return example

def setlabel(imagelabel,lines,w,h):
    for (xmin,ymin,xmax,ymax) in lines:
        for x in range(int(xmin*fixsize_w/w),min(int(xmax*fixsize_w/w),fixsize_w)):
            for y in range(int(ymin*fixsize_h/h),min(int(ymax*fixsize_h/h),fixsize_h)):
                imagelabel[x,y]=255
    return imagelabel

# def dict_to_tf_example_txt(example,
#                            image_subdirectory,
#                            label_path,
#                        label_map_dict=None):
#     """Convert label and file to tf.Example proto.
#
#     Notice that this function normalizes the bounding box coordinates provided
#     by the raw data.
#
#     Args:
#       example: image filename
#       data: a parsed label file which is a list
#       label_map_dict: A map from string label names to integers ids.
#       image_subdirectory: String specifying subdirectory within the
#         Pascal dataset directory holding the actual image data.
#
#     Returns:
#       example: The converted tf.Example.
#
#     """
#     img_path = os.path.join(image_subdirectory, example)
#     with tf.gfile.GFile(img_path, 'rb') as fid:
#         encoded_jpg = fid.read()
#     encoded_jpg_io = io.BytesIO(encoded_jpg)
#     image = PIL.Image.open(encoded_jpg_io)
#     # encoded_label_io = io.BytesIO(encoded_label)
#     # labelimg = PIL.Image.open(encoded_label_io)
#     if image.format != 'JPEG':
#         raise ValueError('Image format not JPEG')
#     key = hashlib.sha256(encoded_jpg).hexdigest()
#
#     width, height = image.size
#
#     data = parse_label(label_path)
#     imagelabel=np.zeros((width,height),dtype=int)
#     imagelabel=setlabel(imagelabel,data)
#
#     with tf.gfile.GFile(label_path, 'rb') as fid:
#         encoded_label = fid.read()
#     feature_dict = {
#         'image/height': dataset_util.int64_feature(height),
#         'image/width': dataset_util.int64_feature(width),
#         'image/filename': dataset_util.bytes_feature(
#             example.encode('utf8')),
#         'image/source_id': dataset_util.bytes_feature(
#             example.encode('utf8')),
#         'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
#         'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#         'image/label/encoded':dataset_util.bytes_feature(encoded_label),
#     }
#     example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
#     return example
def parse_label(label_path):
    '''
    return the parsed label in list manner
    in which one element is a 2-element tuple
    first one is its name (str)
    second one is its loc which is a tuple
    '''
    ret = []
    lines = []
    with open(label_path, 'r') as rfile:
        lines = rfile.readlines()
    for eachline in lines:
        items = eachline.strip().split(' ')
        ret.append((int(float(items[0])), int(float(items[1])), int(float(items[2])), int(float(items[3]))))
    return ret


def create_tf_record(output_filename,
                     label_dir,
                     image_dir,
                     examples,
                     label_examples,
                     label_map_dict=None):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      label_dir: Directory where label files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
      faces_only: If True, generates bounding boxes for pet faces.  Otherwise
        generates bounding boxes (as well as segmentations for full pet bodies).
    """

    # if not os.path.exists(output_filename):
    #     os.makedirs(output_filename)
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx,(example,labelexample) in enumerate(zip(examples,label_examples)):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))

        if FLAGS.format=='txtlabel':
           label_path = os.path.join(label_dir, str(idx).zfill(5)+'frame_groundtruth' + '.txt')

           if not os.path.exists(label_path):
              logging.warning('Could not find %s, ignoring example.', label_path)
              continue

           # data = parse_label(label_path)
           tf_example = dict_to_tf_example(example,labelexample,
                                           image_dir, label_dir,idx)

        # try:
        # tf_example = dict_to_tf_example(
        #     example, data, label_map_dict, image_dir)
        elif FLAGS.format=='imagelabel':
            tf_example = dict_to_tf_example(example,
             labelexample,image_dir,label_dir,idx)
            pass
        else:
            logging.warning('Could not find label format %s, ignoring example.',FLAGS.format)
            continue

        #######writer.write(tf_example.SerializeToString())
        # except ValueError:
        #  logging.warning('Invalid example: %s, ignoring.', xml_path)

    writer.close()
    saveimgpath=os.path.join(os.path.join(image_dir,'..'),'resizedimages')



    # os.system('ffmpeg -i '+saveimgpath+'\\img_%05d.jpg -s '+str(fixsize_w)+'x'+str(fixsize_h)+' -pix_fmt yuvj420p '+saveimgpath+'\\..\\output.yuv')
    # os.chdir(os.path.join(image_dir,'..'))
    # os.system('C:\\Users\\Administrator\\Desktop\\HM-16.7\\bin\\vc2013\\Win32\\Debug\\TAppEncoder -c D:\\encoder_lowdelay_P_main.cfg')
    # data_result=os.path.join(image_dir,'..','data_dir')
    # if not os.path.exists(data_result):
    #     os.makedirs(data_result)
    # os.chdir(data_result)
    # os.system('C:\\Users\\Administrator\\Desktop\\HM-16.7\\bin\\vc2013\\Win32\\Debug\\TAppDecoder -b ../str.bin -o dec.yuv')





# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    # label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Pet dataset.')
    sequence_list=os.listdir(data_dir)
    for file in sequence_list:
      image_dir = os.path.join(data_dir,file, 'input')
      label_dir = os.path.join(data_dir,file, 'groundtruth')
      examples_list = os.listdir(image_dir)
      examples_list = list(map(lambda x: os.path.basename(x), examples_list))
      label_examples=os.listdir(label_dir)
      label_examples=list(map(lambda x: os.path.basename(x), label_examples))

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.

    #   random.seed(42)
    #   random.shuffle(examples_list)
    #   num_examples = len(examples_list)
    #   num_train = int(0.8 * num_examples)
    #   train_examples = examples_list[:num_train]
    #   val_examples = examples_list[num_train:]
    #   logging.info('%d training and %d validation examples.',
    #              len(train_examples), len(val_examples))
      path=os.path.join(FLAGS.output_dir,file)
      if not os.path.exists(path):
          os.makedirs(path)
      train_output_path = os.path.join(path, 'pet_train.record')
      # val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')

      create_tf_record(train_output_path,label_dir,
                     image_dir,examples_list,label_examples)
      # create_tf_record(val_output_path, label_map_dict, label_dir,
      #                image_dir, val_examples)


if __name__ == '__main__':
    tf.app.run()
