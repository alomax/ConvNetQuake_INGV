# -------------------------------------------------------------------
# File Name : models.py
# Creation Date : 11-27-16
# Last Modified : Fri Jan  6 13:38:15 2017
# Author: Thibaut Perol & Michael Gharbi <tperol@g.harvard.edu>
# -------------------------------------------------------------------
"""ConvNetQuake model.
the function get is implemented to help prototyping other models.
One can create a subclass
class Proto(ConvNetQuake)
and overwrite the _setup_prediction method to change the network
architecture
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import tflib.model
import tflib.layers as layers


def get(model_name, inputs, config, checkpoint_dir, is_training=False):
  """Returns a Model instance instance by model name.

  Args:
    name: class name of the model to instantiate.
    inputs: inputs as returned by a DataPipeline.
    params: dict of model parameters.
  """

  return globals()[model_name](inputs, config, checkpoint_dir, is_training=is_training)



class ConvNetQuake(tflib.model.BaseModel):
        
    def __init__(self, inputs, config, checkpoint_dir, n_channels=32, n_conv_layers=8, n_fc_layers=2, is_training=False,  reuse=False):
        
        self.n_channels = n_channels
        self.n_conv_layers = n_conv_layers
        self.n_fc_layers = n_fc_layers
        self.is_training = is_training
        self.cfg = config
        super(ConvNetQuake, self).__init__(inputs, checkpoint_dir, is_training=is_training, reuse=reuse)

    def _setup_prediction(self):
        
        self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    
        current_layer = self.inputs['data']
        #n_channels = 32  # number of channels per conv layer
        ksize = 3  # size of the convolution kernel
        for i in range(self.n_conv_layers):
            current_layer = layers.conv1(current_layer, self.n_channels, ksize, stride=2, scope='conv{}'.format(i + 1), padding='SAME')
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
            self.layers['conv{}'.format(i + 1)] = current_layer
    
        bs, width, _ = current_layer.get_shape().as_list()
        n_fc_nodes = width * self.n_channels
        current_layer = tf.reshape(current_layer, [bs, n_fc_nodes], name="reshape")
        
        # 20180916 AJL - include stram_max in fc layers
        stream_max_tensor = tf.expand_dims(self.inputs['stream_max'], 1)
        current_layer = tf.concat([current_layer, stream_max_tensor], 1)
        n_fc_nodes += 1
    
        for i in range(self.n_fc_layers - 1):
            current_layer = layers.fc(current_layer, n_fc_nodes, scope='fc{}'.format(i + 1), activation_fn=tf.nn.relu)
        current_layer = layers.fc(current_layer, self.cfg.n_distances + self.cfg.n_magnitudes + self.cfg.n_depths + self.cfg.n_azimuths, \
                                  scope='logits', activation_fn=None)

        istart = 0
        iend = self.cfg.n_distances
        self.layers['distance_logits'] = current_layer[ : , istart : iend]
        self.layers['distance_prob'] = tf.nn.softmax(self.layers['distance_logits'], name='distance_prob')
        self.layers['distance_prediction'] = tf.argmax(self.layers['distance_prob'], 1, name='distance_pred')
        istart = iend
        
        self.layers['magnitude_logits'] = tf.constant(-1)
        self.layers['magnitude_prob'] = tf.constant(-1)
        self.layers['magnitude_prediction'] = tf.constant(-1)
        if self.cfg.n_magnitudes > 0:
            iend += self.cfg.n_magnitudes
            self.layers['magnitude_logits'] = current_layer[ : , istart : iend]
            self.layers['magnitude_prob'] = tf.nn.softmax(self.layers['magnitude_logits'], name='magnitude_prob')
            self.layers['magnitude_prediction'] = tf.argmax(self.layers['magnitude_prob'], 1, name='magnitude_pred')
            istart = iend
            
        self.layers['depth_logits'] = tf.constant(-1)
        self.layers['depth_prob'] = tf.constant(-1)
        self.layers['depth_prediction'] = tf.constant(-1)
        if self.cfg.n_depths > 0:
            iend += self.cfg.n_depths
            self.layers['depth_logits'] = current_layer[ : , istart : iend]
            self.layers['depth_prob'] = tf.nn.softmax(self.layers['depth_logits'], name='depth_prob')
            self.layers['depth_prediction'] = tf.argmax(self.layers['depth_prob'], 1, name='depth_pred')
            istart = iend
   
        self.layers['azimuth_logits'] = tf.constant(-1)
        self.layers['azimuth_prob'] = tf.constant(-1)
        self.layers['azimuth_prediction'] = tf.constant(-1)
        if self.cfg.n_azimuths > 0:
            iend += self.cfg.n_azimuths
            self.layers['azimuth_logits'] = current_layer[ : , istart : iend]
            self.layers['azimuth_prob'] = tf.nn.softmax(self.layers['azimuth_logits'], name='azimuth_prob')
            self.layers['azimuth_prediction'] = tf.argmax(self.layers['azimuth_prob'], 1, name='azimuth_pred')
            istart = iend

    
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
    
        tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(self.cfg.regularization),
            weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))

    def validation_metrics(self):
        if not hasattr(self, '_validation_metrics'):
          self._setup_loss()
    
          self._validation_metrics = {
            'loss': self.loss,
            'distance_loss': self.distance_loss,
            'magnitude_loss': self.magnitude_loss,
            'depth_loss': self.depth_loss,
            'azimuth_loss': self.azimuth_loss,
            'detection_accuracy': self.detection_accuracy,
            'distance_accuracy': self.distance_accuracy,
            'magnitude_accuracy': self.magnitude_accuracy,
            'depth_accuracy': self.depth_accuracy,
            'azimuth_accuracy': self.azimuth_accuracy
          }
        return self._validation_metrics

    def validation_metrics_message(self, metrics):
        s = 'loss = {:.5f} | acc = det:{:.1f}% dist:{:.1f}% mag:{:.1f}% dep:{:.1f}% az:{:.1f}%'.format(metrics['loss'],
         metrics['detection_accuracy'] * 100, metrics['distance_accuracy'] * 100, \
         metrics['magnitude_accuracy'] * 100, metrics['depth_accuracy'] * 100, metrics['azimuth_accuracy'] * 100)
        return s
    
    def _setup_loss(self):
        with tf.name_scope('loss'):
            # distance loss
            # change target range from -1:n_distances-1 to 0:n_distances
            targets_distance_id = tf.add(self.inputs['distance_id'], self.cfg.add)
            self.distance_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layers['distance_logits'], labels=targets_distance_id))
            raw_loss = self.distance_loss
            # magnitude loss
            targets_magnitude_id = self.inputs['magnitude_id']
            self.magnitude_loss = tf.constant(0.0)
            if self.cfg.n_magnitudes > 0:
                    vloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layers['magnitude_logits'], labels=targets_magnitude_id)
                    vloss = tf.where(tf.greater(targets_distance_id, tf.zeros_like(targets_distance_id)), vloss, tf.zeros_like(vloss))                    
                    self.magnitude_loss = tf.reduce_mean(vloss)
                    raw_loss += self.magnitude_loss
            # depth loss
            targets_depth_id = self.inputs['depth_id']
#             # TEMPORARY BUG FIX
#             t_clip = tf.clip_by_value(targets_depth_id, 0, self.cfg.n_depths -1)
#             targets_depth_id = t_clip
#             # END TEMPORARY BUG FIX
            self.depth_loss = tf.constant(0.0)
            if self.cfg.n_depths > 0:
                    vloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layers['depth_logits'], labels=targets_depth_id)
                    vloss = tf.where(tf.greater(targets_distance_id, tf.zeros_like(targets_distance_id)), vloss, tf.zeros_like(vloss))                    
                    self.depth_loss = tf.reduce_mean(vloss)
                    raw_loss += self.depth_loss
            # azimuth loss
            targets_azimuth_id = self.inputs['azimuth_id']
            self.azimuth_loss = tf.constant(0.0)
            if self.cfg.n_azimuths > 0:
                    vloss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layers['azimuth_logits'], labels=targets_azimuth_id)
                    vloss = tf.where(tf.greater(targets_distance_id, tf.zeros_like(targets_distance_id)), vloss, tf.zeros_like(vloss))                    
                    self.azimuth_loss = tf.reduce_mean(vloss)
                    raw_loss += self.azimuth_loss
            #
            self.summaries.append(tf.summary.scalar('loss/train_raw', raw_loss))
    
        self.loss = raw_loss
    
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_losses:
          with tf.name_scope('regularizers'):
            reg_loss = sum(reg_losses)
            self.summaries.append(tf.summary.scalar('loss/regularization', reg_loss))
          self.loss += reg_loss
    
        self.summaries.append(tf.summary.scalar('loss/train', self.loss))
    
        with tf.name_scope('accuracy'):
            # change target range from -1:n_distances-1 to 0:n_distances
            targets = tf.add(self.inputs['distance_id'], self.cfg.add)
            is_true_event_logical = tf.greater(targets, tf.zeros_like(targets))
            is_true_event = tf.cast(is_true_event_logical, tf.int64)
            is_pred_event = tf.cast(tf.greater(self.layers['distance_prediction'], tf.zeros_like(targets)), tf.int64)
            detection_is_correct = tf.equal(is_true_event, is_pred_event)
            self.detection_accuracy = tf.reduce_mean(tf.to_float(detection_is_correct))
            self.summaries.append(tf.summary.scalar('detection_accuracy/train', self.detection_accuracy))
            #
            self.distance_accuracy = tf.reduce_mean(tf.to_float(tf.boolean_mask(tf.equal(self.layers['distance_prediction'], targets), is_true_event_logical)))
            #self.distance_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.layers['distance_prediction'], targets)))
            self.summaries.append(tf.summary.scalar('distance_accuracy/train', self.distance_accuracy))
            self.magnitude_accuracy = tf.reduce_mean(tf.to_float(tf.boolean_mask(tf.equal(self.layers['magnitude_prediction'], self.inputs['magnitude_id']), is_true_event_logical)))
            #self.magnitude_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.layers['magnitude_prediction'], self.inputs['magnitude_id'])))
            self.summaries.append(tf.summary.scalar('magnitude_accuracy/train', self.magnitude_accuracy))
            self.depth_accuracy = tf.reduce_mean(tf.to_float(tf.boolean_mask(tf.equal(self.layers['depth_prediction'], self.inputs['depth_id']), is_true_event_logical)))
            #self.depth_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.layers['depth_prediction'], self.inputs['depth_id'])))
            self.summaries.append(tf.summary.scalar('depth_accuracy/train', self.depth_accuracy))
            self.azimuth_accuracy = tf.reduce_mean(tf.to_float(tf.boolean_mask(tf.equal(self.layers['azimuth_prediction'], self.inputs['azimuth_id']), is_true_event_logical)))
            #self.azimuth_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.layers['azimuth_prediction'], self.inputs['azimuth_id'])))
            self.summaries.append(tf.summary.scalar('azimuth_accuracy/train', self.azimuth_accuracy))

    def _setup_optimizer(self, learning_rate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
          updates = tf.group(*update_ops, name='update_ops')
          with tf.control_dependencies([updates]):
            self.loss = tf.identity(self.loss)
        optim = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss, name='optimizer', global_step=self.global_step)
        self.optimizer = optim

    def _tofetch(self):
        return {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'distance_loss': self.distance_loss,
            'magnitude_loss': self.magnitude_loss,
            'depth_loss': self.depth_loss,
            'azimuth_loss': self.azimuth_loss,
            'detection_accuracy': self.detection_accuracy,
            'distance_accuracy': self.distance_accuracy,
            'magnitude_accuracy': self.magnitude_accuracy,
            'depth_accuracy': self.depth_accuracy,
            'azimuth_accuracy': self.azimuth_accuracy
        }

    def _summary_step(self, step_data):
        step = step_data['step']
        loss = step_data['loss']
        distance_loss = step_data['distance_loss']
        magnitude_loss = step_data['magnitude_loss']
        depth_loss = step_data['depth_loss']
        azimuth_loss = step_data['azimuth_loss']
        det_accuracy = step_data['detection_accuracy']
        dist_accuracy = step_data['distance_accuracy']
        mag_accuracy = step_data['magnitude_accuracy']
        dep_accuracy = step_data['depth_accuracy']
        az_accuracy = step_data['azimuth_accuracy']
        duration = step_data['duration']
        avg_duration = 1000 * duration / step
    
        if self.is_training:
          toprint = 'Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} (dist:{:.2f} mag:{:.2f} dep:{:.2f} az:{:.2f}) | acc = det:{:.1f}% d:{:.1f}% m:{:.1f}% dp:{:.1f}% az:{:.1f}%'.format(
            step, duration, avg_duration, loss, distance_loss, magnitude_loss, depth_loss, azimuth_loss, \
            100 * det_accuracy, 100 * dist_accuracy, 100 * mag_accuracy, 100 * dep_accuracy, 100 * az_accuracy)
        else:
          toprint = 'Step {} | {:.0f}s ({:.0f}ms) | acc = det:{:.1f}% dist:{:.1f}% mag:{:.1f}% dep:{:.1f}% az:{:.1f}%'.format(
            step, duration, avg_duration, 100 * det_accuracy, 100 * dist_accuracy, 100 * mag_accuracy, 100 * dep_accuracy, 100 * az_accuracy)
    
        return toprint



class ConvNetQuake7(ConvNetQuake):
    
    def __init__(self, inputs, config, checkpoint_dir, is_training=False, reuse=False):
        
        n_conv_layers = 7
        super(ConvNetQuake7, self).__init__(inputs, config, checkpoint_dir, n_conv_layers=n_conv_layers, is_training=is_training,
                                        reuse=reuse)


class ConvNetQuake6(ConvNetQuake):
    
    def __init__(self, inputs, config, checkpoint_dir, is_training=False, reuse=False):
        
        n_conv_layers = 6
        super(ConvNetQuake6, self).__init__(inputs, config, checkpoint_dir, n_conv_layers=n_conv_layers, is_training=is_training,
                                        reuse=reuse)
    
class ConvNetQuake9(ConvNetQuake):
    
    def __init__(self, inputs, config, checkpoint_dir, is_training=False, reuse=False):
        
        n_conv_layers = 9
        super(ConvNetQuake9, self).__init__(inputs, config, checkpoint_dir, n_conv_layers=n_conv_layers, is_training=is_training,
                                        reuse=reuse)
    
    
class ConvNetQuake_ch64_cv8_fc4(ConvNetQuake):
    
    def __init__(self, inputs, config, checkpoint_dir, is_training=False, reuse=False):
        
        n_channels=64
        n_conv_layers=8
        n_fc_layers=4
        super(ConvNetQuake_ch64_cv8_fc4, self).__init__(inputs, config, checkpoint_dir, 
                                                     n_channels=n_channels, n_conv_layers=n_conv_layers, n_fc_layers= n_fc_layers, is_training=is_training, 
                                                     reuse=reuse)
    
    
class ConvNetQuake_ch32_cv8_fc4(ConvNetQuake):
    
    def __init__(self, inputs, config, checkpoint_dir, is_training=False, reuse=False):
        
        n_channels=32
        n_conv_layers=8
        n_fc_layers=4
        super(ConvNetQuake_ch32_cv8_fc4, self).__init__(inputs, config, checkpoint_dir, 
                                                     n_channels=n_channels, n_conv_layers=n_conv_layers, n_fc_layers= n_fc_layers, is_training=is_training, 
                                                     reuse=reuse)
    
    
    
    
    
    
    
    
    
