'''
Created on 29 Mar 2018

@author: Anthony Lomax - ALomax Scientific
'''
from h5py._hl import dataset

"""Get streams from FDSN web service."""

import os
import argparse
import cPickle as pickle

from shutil import copy

import numpy as np

from obspy import read

from quakenet.data_pipeline import DataWriter

import tensorflow as tf





def main(args):
    
    
    if not os.path.exists(args.outroot):
        os.makedirs(args.outroot)
        
    # copy some files
    copy(os.path.join(args.inroot, 'params.pkl'), args.outroot)
    copy(os.path.join(args.inroot, 'event_channel_dict.pkl'), args.outroot)
        
    if not os.path.exists(args.outroot):
        os.makedirs(args.outroot)   

    for dataset in ['train', 'validate', 'test']:
        for datatype in ['events', 'noise']:
            inpath = os.path.join(args.inroot, dataset, datatype)
            outpath = os.path.join(args.outroot, dataset, datatype)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            mseedpath = os.path.join(outpath, 'mseed')
            if not os.path.exists(mseedpath):
                os.makedirs(mseedpath)
            mseedpath = os.path.join(outpath, 'mseed_raw')
            if not os.path.exists(mseedpath):
                os.makedirs(mseedpath)
            if datatype == 'events':
                xmlpath = os.path.join(outpath, 'xml')
                if not os.path.exists(xmlpath):
                    os.makedirs(xmlpath)

            # inroot example: output/MN/streams
            # inpath example: output/MN/streams/train/events
            for dirpath, dirnames, filenames in os.walk(inpath):
                for name in filenames:
                    if name.endswith(".tfrecords"):
                        filename_root = name.replace('.tfrecords', '')
                        print 'Processing:', name, os.path.join(outpath, filename_root + '.tfrecords')
                        
                        # copy some files
                        copy(os.path.join(inpath, 'mseed_raw', filename_root + '.mseed'), os.path.join(outpath, 'mseed_raw'))
                        if datatype == 'events':
                            copy(os.path.join(inpath, 'xml', filename_root + '.xml'), os.path.join(outpath, 'xml'))

                        # read raw mseed
                        stream = read(os.path.join(inpath, 'mseed_raw', filename_root + '.mseed'), format='MSEED')
                        # store absolute maximum
                        stream_max = np.absolute(stream.max()).max()
                        # normalize by absolute maximum
                        stream.normalize(global_max = True)
                        # write new processed miniseed
                        streamfile = os.path.join(outpath, 'mseed', filename_root + '.mseed')
                        stream.write(streamfile, format='MSEED', encoding='FLOAT32')

                        n_traces = 3
                        win_size = 10001

                        # read old tfrecords
                        # https://www.kaggle.com/mpekalski/reading-tfrecord
                        record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(inpath, filename_root + '.tfrecords'))
                        for string_record in record_iterator:
                            example = tf.train.Example()
                            example.ParseFromString(string_record)
                            distance_id = int(example.features.feature['distance_id'].int64_list.value[0])                            
                            magnitude_id = int(example.features.feature['magnitude_id'].int64_list.value[0])                            
                            depth_id = int(example.features.feature['depth_id'].int64_list.value[0])                            
                            azimuth_id = int(example.features.feature['azimuth_id'].int64_list.value[0])                            
                            distance = float(example.features.feature['distance'].float_list.value[0])                            
                            magnitude = float(example.features.feature['magnitude'].float_list.value[0])                            
                            depth = float(example.features.feature['depth'].float_list.value[0])                            
                            azimuth = float(example.features.feature['azimuth'].float_list.value[0])                            
                            print 'id', distance_id, 'im', magnitude_id, 'ide', depth_id, 'iaz', azimuth_id, \
                                'd', distance, 'm', magnitude, 'de', depth, 'az', azimuth
                        
#                         filename_queue = tf.train.string_input_producer([os.path.join(inpath, filename_root + '.tfrecords')], shuffle=False)
#                         reader = tf.TFRecordReader()
#                         example_key, serialized_example = reader.read(filename_queue)
#                         # data_pipeline._parse_example()
#                         features = tf.parse_single_example(
#                             serialized_example,
#                             features={
#                             'window_size': tf.FixedLenFeature([], tf.int64),
#                             'n_traces': tf.FixedLenFeature([], tf.int64),
#                             'data': tf.FixedLenFeature([], tf.string),
#                             #'stream_max': tf.FixedLenFeature([], tf.float32),
#                             'distance_id': tf.FixedLenFeature([], tf.int64),
#                             'magnitude_id': tf.FixedLenFeature([], tf.int64),
#                             'depth_id': tf.FixedLenFeature([], tf.int64),
#                             'azimuth_id': tf.FixedLenFeature([], tf.int64),
#                             'distance': tf.FixedLenFeature([], tf.float32),
#                             'magnitude': tf.FixedLenFeature([], tf.float32),
#                             'depth': tf.FixedLenFeature([], tf.float32),
#                             'azimuth': tf.FixedLenFeature([], tf.float32),
#                             'start_time': tf.FixedLenFeature([],tf.int64),
#                             'end_time': tf.FixedLenFeature([], tf.int64)})
#                         features['name'] = example_key
#                         # END - data_pipeline._parse_example()
# 
#                         print "features['distance_id']", features['distance_id']
#                         print 'distance_id shape', tf.shape(features['distance_id'])
#                         with tf.Session() as sess:
#                             print sess.run(features['distance_id'])
#                         #print 'distance_id', distance_id
#                         #print 'distance_id shape', tf.shape(distance_id)
#                         magnitude_id = features['magnitude_id']
#                         depth_id = features['depth_id']
#                         azimuth_id = features['azimuth_id']
#                         distance = features['distance']
#                         magnitude = features['magnitude']
#                         depth = features['depth']
#                         azimuth = features['azimuth']                        

                        # write new tfrecords
                        writer = DataWriter(os.path.join(outpath, filename_root + '.tfrecords'))
                        writer.write(stream, stream_max, distance_id, magnitude_id, depth_id, azimuth_id, distance, magnitude, depth, azimuth)

                    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--inroot', type=str,
                        help='Path for input')
    parser.add_argument('--outroot', type=str,
                        help='Path for output')


    args = parser.parse_args()

    main(args)
