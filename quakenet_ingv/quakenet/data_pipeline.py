"""Classes and functions to read, write and feed data."""

import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import csv
import json
from obspy.core.utcdatetime import UTCDateTime

POSITIVE_EXAMPLES_PATH = 'positive'
NEGATIVE_EXAMPLES_PATH = 'negative'
# RECORD_REGEXP = re.compile(r'\d+\.tfrecords')


class DataWriter(object):

    """ Writes .tfrecords file to disk from window Stream objects.
    """

    def __init__(self, filename):
        self._writer = None
        self._filename = filename
        self._written = 0
        self._writer = tf.python_io.TFRecordWriter(self._filename)

    def write(self, sample_window, stream_max, distance_id, magnitude_id, depth_id, azimuth_id, distance, magnitude, depth, azimuth):
        n_traces = len(sample_window)
        n_samples = len(sample_window[0].data)
        # Extract data
        data = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            data[i, :] = sample_window[i].data[...]
        # Extract metadata
        start_time = np.int64(sample_window[0].stats.starttime.timestamp)
        end_time = np.int64(sample_window[0].stats.endtime.timestamp)
        # print('starttime {}, endtime {}'.format(UTCDateTime(start_time),
                                                # UTCDateTime(end_time)))

        example = tf.train.Example(features=tf.train.Features(feature={
            'window_size': self._int64_feature(n_samples),
            'n_traces': self._int64_feature(n_traces),
            'data': self._bytes_feature(data.tobytes()),
            'stream_max': self._float_feature(stream_max),
            'distance_id': self._int64_feature(distance_id),
            'magnitude_id': self._int64_feature(magnitude_id),
            'depth_id': self._int64_feature(depth_id),
            'azimuth_id': self._int64_feature(azimuth_id),
            'distance': self._float_feature(distance),
            'magnitude': self._float_feature(magnitude),
            'depth': self._float_feature(depth),
            'azimuth': self._float_feature(azimuth),
            'start_time': self._int64_feature(start_time),
            'end_time': self._int64_feature(end_time),
        }))
        self._writer.write(example.SerializeToString())
        self._written += 1

    def close(self):
        self._writer.close()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataReader(object):

    def __init__(self, path, config, shuffle=True):
        self._path = path
        self._shuffle = shuffle
        self._config = config
        self.win_size = config.win_size
        self.n_traces = config.n_traces


        self._reader = tf.TFRecordReader()

    def read(self):
        filename_queue = self._filename_queue()
        example_key, serialized_example = self._reader.read(filename_queue)
        example = self._parse_example(example_key, serialized_example)
        return example

    def _filename_queue(self):
        fnames = []
        for root, dirs, files in os.walk(self._path):
            for f in files:
                if f.endswith(".tfrecords"):
                    if self._config.unique_station is None or self._config.unique_station in f:
                        fnames.append(os.path.join(root, f))
        fname_q = tf.train.string_input_producer(fnames,
                                                 shuffle=self._shuffle,
                                                 num_epochs=self._config.n_epochs)
        return fname_q

    def _parse_example(self, example_key, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'window_size': tf.FixedLenFeature([], tf.int64),
                'n_traces': tf.FixedLenFeature([], tf.int64),
                'data': tf.FixedLenFeature([], tf.string),
                'stream_max': tf.FixedLenFeature([], tf.float32),
                'distance_id': tf.FixedLenFeature([], tf.int64),
                'magnitude_id': tf.FixedLenFeature([], tf.int64),
                'depth_id': tf.FixedLenFeature([], tf.int64),
                'azimuth_id': tf.FixedLenFeature([], tf.int64),
                'distance': tf.FixedLenFeature([], tf.float32),
                'magnitude': tf.FixedLenFeature([], tf.float32),
                'depth': tf.FixedLenFeature([], tf.float32),
                'azimuth': tf.FixedLenFeature([], tf.float32),
                'start_time': tf.FixedLenFeature([],tf.int64),
                'end_time': tf.FixedLenFeature([], tf.int64)})

        # Convert and reshape
        data = tf.decode_raw(features['data'], tf.float32)
        data.set_shape([self.n_traces * self.win_size])
        data = tf.reshape(data, [self.n_traces, self.win_size])
        data = tf.transpose(data, [1, 0])

        # Pack
        features['data'] = data

        features['name'] = example_key

        return features


class DataPipeline(object):

    """Creates a queue op to stream data for training.

    Attributes:
    samples: Tensor(float). batch of input samples [batch_size, n_channels, n_points]
    labels: Tensor(int32). Corresponding batch 0 or 1 labels, [batch_size,]

    """

    def __init__(self, dataset_path, config, is_training):

        min_after_dequeue = 1000
        capacity = 1000 + 3 * config.batch_size

        if is_training:

            with tf.name_scope('inputs'):
                self._reader = DataReader(dataset_path, config=config)
                samples = self._reader.read()
                tensors = [samples['name'], samples['data'], samples['stream_max'], samples["distance_id"], samples["magnitude_id"], samples["depth_id"], samples["azimuth_id"]]

                self.names, self.samples, self.stream_max, self.distance_id, self.magnitude_id, self.depth_id, self.azimuth_id = tf.train.shuffle_batch(
                    tensors,
                    batch_size=config.batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    allow_smaller_final_batch=False)

        elif not is_training:

            with tf.name_scope('validation_inputs'):
                self._reader = DataReader(dataset_path, config=config)
                samples = self._reader.read()

                tensors = [samples['name'], samples['data'], samples['stream_max'], 
                           samples["distance_id"], samples["magnitude_id"], samples["depth_id"], samples["azimuth_id"], 
                           samples["distance"], samples["magnitude"], samples["depth"], samples["azimuth"], 
                           samples["start_time"], samples["end_time"]]

                self.names, self.samples, self.stream_max, self.distance_id, self.magnitude_id, self.depth_id, self.azimuth_id, \
                    self.distance, self.magnitude, self.depth, self.azimuth, self.start_time, self.end_time = tf.train.batch(
                    tensors,
                    batch_size=config.batch_size,
                    capacity=capacity,
                    num_threads=config.n_threads,
                    allow_smaller_final_batch=False)
        else:
            raise ValueError(
                "is_training flag is not defined, set True for training and False for testing")

