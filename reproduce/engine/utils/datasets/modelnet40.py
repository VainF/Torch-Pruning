#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py

@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download(path):
    DATA_DIR = path
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('mv %s %s' % (zipfile, DATA_DIR))


def load_data(data_root, partition, download_path='data'):
    download(download_path)
    
    DATA_DIR = data_root
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, data_root, num_points, partition='train'):
        self.data, self.label = load_data(data_root, partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        pointcloud = np.transpose(pointcloud, (1, 0))
        label = np.squeeze(label)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)