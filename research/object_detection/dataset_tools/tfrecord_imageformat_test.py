# # -*- coding: utf-8 -*-
# # @Time    : 5/18/2018 4:27 PM
# # @Author  : v-weyu
# # @Email   : v-weyu@microsoft.com
# # @File    : read_record_test.py
# # @Software: PyCharm
# import tensorflow as tf
# from PIL import Image
# # from object_detection  import trainer
# from research.object_detection.data_decoders import  tf_example_decoder as te
# import matplotlib.pyplot as plt
# # from object_detection.meta_architectures import ssd_meta_arch
# # tfrecords_filename = "D:/v-weyu/work/tf-logoRec/data_viewsize/pet_train_viewsize_new.record"
# tfrecords_filename = "D:/video_data/dataset2014cdw/record/backdoor/pet_train.record"
# # tfrecords_filename = "D:/v-weyu/logo-rec/tf-logoRec/data_viewsize/data/pet_train_viewsize.record"
# # test_write_to_tfrecords(tfrecords_filename)
#
# # filename_queue = tf.train.string_input_producer([tfrecords_filename], )  # 读入流中
# # reader = tf.TFRecordReader()
# # _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
# # decoder = te.TfExampleDecoder(
# #     label_map_proto_file="D:/v-weyu/work/tf-logoRec/data_viewsize/label_map.pbtxt")
# # features =decoder.decode(serialized_example)
#
# parallel_reader = tf.contrib.slim.parallel_reader
# _, string_tensor = parallel_reader.parallel_read(
#     tfrecords_filename,  # Convert `RepeatedScalarContainer` to list.
#     reader_class=tf.TFRecordReader,
#     num_epochs=None,
#     num_readers=8,
#     shuffle=False,
#     dtypes=[tf.string, tf.string],
#     capacity=200,
#     min_after_dequeue=5)
# decoder = te.TfExampleDecoder(
#         # load_instance_masks="D:/v-weyu/work/tf-logoRec/data_viewsize/label_map.pbtxt")
# label_map_proto_file="D:/v-weyu/work/tf-logoRec/data_viewsize/label_map.pbtxt")
# features=decoder.decode(string_tensor)
# image=features['image']
# filename=features['filename']
# groundtruth_area=features['groundtruth_area']
# groundtruth_boxes=features['groundtruth_boxes']
# groundtruth_classes=features['groundtruth_classes']
# import  os
# from object_detection.core import batcher
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# batch_size = 5
# mini_after_dequeue = 100
# capacity = mini_after_dequeue+3*batch_size
#
#
# # def get_batch_tensor():
#
# # example_batch,filename_batch,label_box_batch,label_classes_batch = tf.train.batch([image,filename,groundtruth_boxes,groundtruth_classes],batch_size = batch_size,capacity=capacity)
# input_queue = batcher.BatchQueue(
#       features,
#       batch_size=batch_size,
#       batch_queue_capacity=300,
#       num_batch_queue_threads=8,
#       prefetch_queue_capacity=5)
# batch_tensor=input_queue.dequeue()
#
# # dic=dict()
# # const1=tf.constant([0,2,3,4])
# # l=[]
# # # l.append(tf.constant(12))
# # # l.append(tf.constant(13))
# # # l.append(const1)
# # l2=[]
# # l2.append(tf.constant('test'))
# # l3=[]
# # l3.append(tf.constant('string'))
# # l4=[]
# # l4.append(tf.constant('l4'))
# # l.append(const1)
# # l.append(l2)
# # l.append(l3)
# # l.append(l4)
# # li=list(l)
# # l2_list=list(l2)
# # l3_list=list(l3)
# # l4_list=list(l4)
# # li_=[]
# # li_.append(list(l2_list))
# # li_.append(list(l3_list))
# # li_.append(list(l4_list))
# #
# # dic['constant']=const1
# # dic['list']=l
# # dic['li']=li
# # dic['li_']=li_
# # l2.pop(0)
# # l2.append(tf.constant('againlist'))
# # l[0]=tf.constant([[1,2,3],[2,3,4]])
# # # const1=tf.constant(1)
# # # l.append(tf.constant(16))
# # # l.pop(0)
# # # l.__setitem__(2,tf.constant(1))
# imagelist_float=[]
# imagelist=[]
# ori_images=[]
# # meta=ssd_meta_arch.SSDMetaArch()
# def process(inputs):
#     return (2.0 / 255.0) * inputs - 1.0
# for var in batch_tensor:
#     image=var['image']
#     ori_images.append(image)
#     # image=tf.expand_dims(image,0)
#     # image=trainer.resize_image_with_pad(image,tf.constant(1024),tf.constant(1024))
#     imagelist.append(image)
#     image=process(tf.to_float(image))
#     imagelist_float.append(image)
# batchimage=tf.stack(imagelist)
# # con1=tf.constant([1,2,3])
# # con2=tf.constant([2,3])
# # con3=tf.concat([con1,con2],0)
# with tf.Session() as sess:  # 开始一个会话
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(200000000):
#         # example, xmax,xmin,ymax,ymin = sess.run([image, label_xmax,label_xmin,label_ymax,label_ymin])  # 在会话中取出image和label
#         # img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
#         # img.save('./' + str(i) + '_''Label_' + str(l) + '.jpg')  # 存下图片
#         # print(example, xmax,xmin,ymax,ymin)
#
#         # batch_data=sess.run([batch_tensor])
#         # con=sess.run([con3])
#         padimagelist,result_ori_images,floatimg,batch_result=sess.run([imagelist,ori_images,imagelist_float,batch_tensor])
#         # for (image,orig,floatimg) in zip(padimagelist,result_ori_images,floatimg):
#         #     img = Image.fromarray(image, 'RGB')
#         #     # img.save('my.png')
#         #     # img.show()
#         #     img.save('padimg.png')
#         #     img1=Image.fromarray(orig,'RGB')
#         #     # img1.show()
#         #     img1.save('ori.png')
#         #     img2=Image.fromarray(floatimg,'RGB')
#         #     img2.save('floatimg.png')
#         #     # plt.imshow(img2, cmap='gray', clim=(0, 1))
#         #     plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
#         # print(result_padimage)
#         # print(example)
#         # print(file)
#         # print(classes)
#         # print(boxes)
#         # print(batch_data[0])
#
#     coord.request_stop()
#     coord.join(threads)




import tensorflow as tf
import matplotlib.pyplot as plt

# data_path = 'D:/video_data/dataset2014cdw/record/backdoor/pet_train.record'
data_path = 'D:/video_data/record/backdoor'

with tf.Session() as sess:
    # feature key and its data type for data restored in tfrecords file
    feature = {
               'image/height':tf.FixedLenFeature([], tf.int64),
               'image/width':tf.FixedLenFeature([], tf.int64),
               'image/filename':tf.FixedLenFeature([], tf.string),
               'image/source_id':tf.FixedLenFeature([], tf.string),
                'image/key/sha256':tf.FixedLenFeature([], tf.string),
                'image/encoded':tf.FixedLenFeature([], tf.string),
                'image/label/encoded':tf.FixedLenFeature([], tf.string)
                     }
    # define a queue base on input filenames
    # filename_queue = tf.train.string_input_producer([data_path], num_epoches=1,shuffle=False)
    from tensorflow.python.platform import gfile
    import os
    filenames = gfile.Glob(os.path.join(data_path, '*'))
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    # define a tfrecords file reader
    reader = tf.TFRecordReader()
    # read in serialized example data
    _, serialized_example = reader.read(filename_queue)
    # decode example by feature
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # convert dtype from unit8 to float32 for later resize

    labelimage = tf.image.decode_png(features['image/label/encoded'])
    labelimage = tf.image.convert_image_dtype(labelimage, dtype=tf.float32)
    labelimage=tf.squeeze(labelimage)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    # restore image to [height, width, 3]
    image = tf.reshape(image, [height, width, 3])
    # resize
    # image = tf.image.resize_images(image, [224, 224])
    # create bathch
    # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10) # capacity是队列的最大容量，num_threads是dequeue后最小的队列大小，num_threads是进行队列操作的线程数。
    # input_queue = batcher.BatchQueue(
    #           features,
    #           batch_size=1,
    #           batch_queue_capacity=300,
    #           num_batch_queue_threads=8,
    #           prefetch_queue_capacity=5)
    # batch_tensor=input_queue.dequeue()

    # initialize global & local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # create a coordinate and run queue runner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # for batch_index in range(3):
    #     batch_images, batch_labels = sess.run([images, labels])
    #     for i in range(10):
    #         plt.imshow(batch_images[i, ...])
    #         plt.show()
    #         print "Current image label is: ", batch_labels[i]
    for  i in range(1386):
      img,label,h,w=sess.run([image,labelimage,height,width])
      plt.imshow(img)
      # plt.imshow(label, cmap=plt.get_cmap('gray'))
      # plt.cm.gray_r
      # plt.imshow(label, cmap=plt.cm.gray_r)
      img=label
      plt.subplot(221)
      plt.imshow(img, cmap='gray')
      plt.subplot(222)
      plt.imshow(img, cmap=plt.cm.gray)
      plt.subplot(223)
      plt.imshow(img, cmap=plt.cm.gray_r)
      plt.show()
      ind=label>0.5
      ind1=label<=0.5
      label[ind]=1
      label[ind1]=0
      plt.subplot(224)
      plt.imshow(label, cmap='gray')

    # close threads
    coord.request_stop()
    coord.join(threads)
    sess.close()