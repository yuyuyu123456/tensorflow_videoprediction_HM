import os

# data_dir='D:/video_data/pets2009'
# data_dir='D:/video_data/videocaviar'
# sequence_list=os.listdir(data_dir)
# for file in sequence_list:
#       label_dir = os.path.join(data_dir,file, 'groundtruth')
#       examples_list = os.listdir(label_dir)
#       for example in examples_list:
#           newname=example.split('f')[0].zfill(5)+'frame_groundtruth.txt'
#           os.rename(os.path.join(label_dir,example),os.path.join(label_dir,newname))
import struct
from sklearn import preprocessing
from pandas import DataFrame
import numpy as np

minheight=8
minwidth=8
blockheight=8
blockwidth=8
blocksize_width=64
blocksize_height=64
data_len=32
with open('C:/Users/Administrator/Desktop/HM-16.7_加密和提取 - Copy/bin/vc2013/Win32/Debug/00000slice.txt', 'r') as f:
    try:
        # while True:
        #     data_id = struct.unpack("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii", f.read(128))
        #     print(data_id)
        lines=f.readlines()
        # print(str)
        data=np.zeros((blockwidth,blockheight,minwidth,minheight,data_len))
        for line in lines:
            # data[line[0]]=0
            line=line.split(" ")
            print(line)
            blockx=int(int(line[0])/64)
            blocky=int(int(line[1])/64)
            cux=int((int(line[0])-blockx*blocksize_width)/blockwidth)
            cuy=int((int(line[1])-blocky*blocksize_height)/blockheight)
            for x in range(int(int(line[2])/blockwidth)):
                for y in range(int(int(line[3])/blockheight)):
                    for i in range(data_len):
                       data[int(blockx)][int(blocky)][cux+x][cuy+y][i]=int(line[i])
                    print(data[int(blockx)][int(blocky)][cux+x][cuy+y])

    except EOFError:
        pass
    except IOError:
        pass
    except struct.error:
        pass



# import numpy as np
# X = np.array([[ 1., -1.,  2.],
#                [ 2.,  0.,  0.],
#                [ 0.,  1., -1.]])
# X_scaled = preprocessing.scale(X)
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(X)
# print(scaled)
# import pandas as pd
# import numpy as np
#
# data = np.random.randint(0,10000000,size=(3,2,4,5))
# # p = pd.Panel(data)
# print(data)
# p = pd.Panel4D(data)
# print(p)
# try:
#     while True:
#         sent = f.readline()
#         print
#         sent
# except EOFError:
#     pass

