#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adam (adamantios.ntakaris@ed.ac.uk)
"""

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

@keras_export(v1=['keras.layers.LSTMCell'])
class RevisedLSTMCell(DropoutRNNCellMixin, Layer):
  def __init__(self,
               units ,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    if units < 0:
      raise ValueError(f'Received an invalid value for argument `units`, '
                       f'expected a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if tf.compat.v1.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(RevisedLSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    
    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.state_size = [self.units, self.units]
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    
    input_dim = input_shape[-1]-1 
    
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        )
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        )

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return backend.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.get('ones')((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          )
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    
    i = self.recurrent_activation(x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + backend.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c_t = self.activation(x_c + backend.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    c = f * c_tm1 + i * c_t
    o = self.recurrent_activation(x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    
    return c, o, i, f, c_t

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o
  
  def call(self, inputs, states, training=None):

    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state
    
    inputs_1 = inputs[0][0:-1].reshape(1,-1)
    
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs_1 * dp_mask[0]
        inputs_f = inputs_1 * dp_mask[1]
        inputs_c = inputs_1 * dp_mask[2]
        inputs_o = inputs_1 * dp_mask[3]
      else:
        inputs_i = inputs_1 
        inputs_f = inputs_1
        inputs_c = inputs_1
        inputs_o = inputs_1
      k_i, k_f, k_c, k_o = tf.split(
          self.kernel, num_or_size_splits=4, axis=1)
      x_i = backend.dot(inputs_i, k_i)
      x_f = backend.dot(inputs_f, k_f)
      x_c = backend.dot(inputs_c, k_c)
      x_o = backend.dot(inputs_o, k_o)
     
      if self.use_bias:
        b_i, b_f, b_c, b_o = tf.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = backend.bias_add(x_i, b_i)
        x_f = backend.bias_add(x_f, b_f)
        x_c = backend.bias_add(x_c, b_c)
        x_o = backend.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o, i, f, c_t = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs_1 = inputs_1 * dp_mask[0]
      z = backend.dot(inputs_1, self.kernel)
      z += backend.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = backend.bias_add(z, self.bias)

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h_1 = o * self.activation(c)
    
    gated_vector = tf.concat([i, f, c_t, c, o, h_1], axis=1)
      
    gated_vector_copy = tf.tile(gated_vector, (copies,1))

    gated_labels = tf.tile(inputs[0][-1].reshape(1,-1), (copies,1))

    n_epoch = 13
    learning_rate = 0.0001
        
    theta_1 = tf.ones([self.units*6, 1])

    for epoch in range(n_epoch):
        y_pred = tf.matmul(gated_vector_copy, theta_1)


        error = y_pred - gated_labels
        gradients = 2/copies * tf.matmul(tf.transpose(gated_vector_copy), tf.cast(error, tf.float32))
        theta_1 = theta_1 - learning_rate * gradients

    importance = theta_1

  
    i_gate_out = importance[:self.units, :]
    f_gate_out = importance[self.units:self.units*2, :]
    can_gate_out = importance[self.units*2:self.units*3, :]
    c_gate_out = importance[self.units*3:self.units*4, :]
    o_gate_out = importance[self.units*4:self.units*5, :]
    h_gate_out = importance[self.units*5:self.units*6, :]
    
    #importance score
    improtance_i = tf.math.reduce_mean(i_gate_out, axis = 0)
    importance_f = tf.math.reduce_mean(f_gate_out, axis = 0)
    importance_can = tf.math.reduce_mean(can_gate_out, axis = 0) 
    importance_c = tf.math.reduce_mean(c_gate_out, axis = 0)
    importance_o = tf.math.reduce_mean(o_gate_out, axis = 0)
    importance_h = tf.math.reduce_mean(h_gate_out, axis = 0)
   
    # final/optimized ouput
    merge_output = tf.stack([improtance_i, importance_f, importance_can, importance_c, importance_o, importance_h], axis = 0)
    result = tf.where(merge_output == tf.math.reduce_max(merge_output, axis = 0)) 
    
    # best gate filter
    if result[0][0] == 0:
        h = tf.transpose(i_gate_out)
    elif result[0][0] == 1:
        h = tf.transpose(f_gate_out)
    elif result[0][0] == 2:
        h = tf.transpose(can_gate_out)
    elif result[0][0] == 3:
        h = tf.transpose(c_gate_out)
    elif result[0][0] == 4:
        h = tf.transpose(o_gate_out)
    else:
        h = tf.transpose(h_gate_out)
    return h, [h, c]
    

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTMCellTrial, self).get_config() 
    return dict(list(base_config.items()) + list(config.items()))