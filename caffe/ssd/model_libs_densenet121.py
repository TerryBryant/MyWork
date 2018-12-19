from __future__ import print_function
caffe_root = '/home/terry/software/caffe-ssd-gpu'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1, use_group = False,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):

    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=1)],
            'weight_filler': dict(type='msra'),
            'bias_term': False,
            }
        eps = bn_params.get('eps', 1e-5)
        use_global_stats = bn_params.get('use_global_stats', False)
        # parameters for batchnorm layer.
        bn_kwargs = {
            'param': [
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            }
        bn_lr_mult = lr_mult
        if use_global_stats:
            # only specify if use_global_stats is explicitly provided;
            # otherwise, use_global_stats_ = this->phase_ == TEST;
            bn_kwargs = {
              'param': [
                  dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0)],
              'eps': eps,
              'use_global_stats': use_global_stats,
              }
            # not updating scale/bias parameters
            bn_lr_mult = 0
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
              'bias_term': True,
              'param': [
                  dict(lr_mult=bn_lr_mult, decay_mult=0),
                  dict(lr_mult=bn_lr_mult, decay_mult=0)],
              'filler': dict(type='constant', value=1.0),
              'bias_filler': dict(type='constant', value=0.0),
              }
        else:
            bias_kwargs = {
              'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
              'filler': dict(type='constant', value=0.0),
              }
    else:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        if use_group:
            net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, group=num_output,
                kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
        else:
            net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    else:
        if use_group:
            net[conv_name] = L.Convolution(net[from_layer], num_output=num_output, group=num_output,
                kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                stride_h=stride_h, stride_w=stride_w, **kwargs)
        else:
            net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
               kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
               stride_h=stride_h, stride_w=stride_w, **kwargs)
    if dilation > 1:
        net.update(conv_name, {'dilation': dilation})

    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
        sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
        net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
        bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
        net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = 'relu{}'.format(conv_name.strip('conv'))
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


# Pay attention that the structure of densenet is more like the resnet v2
def NewConvBNLayer(net, from_layer, out_layer, use_conv, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1, use_group = False,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):

    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=lr_mult, decay_mult=1)],
            'weight_filler': dict(type='msra'),
            'bias_term': False,
            }
        eps = bn_params.get('eps', 1e-5)
        use_global_stats = bn_params.get('use_global_stats', False)
        # parameters for batchnorm layer.
        bn_kwargs = {
            'param': [
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0),
                dict(lr_mult=0, decay_mult=0)],
            'eps': eps,
            }
        bn_lr_mult = lr_mult
        if use_global_stats:
            # only specify if use_global_stats is explicitly provided;
            # otherwise, use_global_stats_ = this->phase_ == TEST;
            bn_kwargs = {
              'param': [
                  dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0),
                  dict(lr_mult=0, decay_mult=0)],
              'eps': eps,
              'use_global_stats': use_global_stats,
              }
            # not updating scale/bias parameters
            bn_lr_mult = 0
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
              'bias_term': True,
              'param': [
                  dict(lr_mult=bn_lr_mult, decay_mult=0),
                  dict(lr_mult=bn_lr_mult, decay_mult=0)],
              'filler': dict(type='constant', value=1.0),
              'bias_filler': dict(type='constant', value=0.0),
              }
        else:
            bias_kwargs = {
              'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
              'filler': dict(type='constant', value=0.0),
              }
    else:
        kwargs = {
            'param': [
                dict(lr_mult=lr_mult, decay_mult=1),
                dict(lr_mult=2 * lr_mult, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)

    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[from_layer], in_place=False, **bn_kwargs)    # in_place decides whether to have the same input of last layer
    if use_scale:
        sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
        net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
        bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
        net[bias_name] = L.Bias(net[from_layer], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = 'relu{}'.format(conv_name.strip('conv'))
        net[relu_name] = L.ReLU(net[bn_name], in_place=True)

    if use_conv:
        [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
        [pad_h, pad_w] = UnpackVariable(pad, 2)
        [stride_h, stride_w] = UnpackVariable(stride, 2)
        if kernel_h == kernel_w:
            if use_group:
                net[conv_name] = L.Convolution(net[relu_name], num_output=num_output, group=num_output,
                    kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
            else:
                net[conv_name] = L.Convolution(net[relu_name], num_output=num_output,
                    kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
        else:
            if use_group:
                net[conv_name] = L.Convolution(net[relu_name], num_output=num_output, group=num_output,
                    kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                    stride_h=stride_h, stride_w=stride_w, **kwargs)
            else:
                net[conv_name] = L.Convolution(net[relu_name], num_output=num_output,
                   kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                   stride_h=stride_h, stride_w=stride_w, **kwargs)
        if dilation > 1:
            net.update(conv_name, {'dilation': dilation})


def DenseBlock(net, from_layer, out_layer, dilation=1, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = ''
    bn_postfix = '/bn'
    scale_prefix = ''
    scale_postfix = '/scale'
    use_scale = True

    out_name = out_layer + '/x1'

    NewConvBNLayer(net, from_layer, out_name, use_conv=True, use_bn=True, use_relu=True,
                   num_output=128, kernel_size=1, pad=0, stride=1,
                   dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                   bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                   scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    out_name = out_layer + '/x2'

    if dilation == 1:
        NewConvBNLayer(net, net.keys()[-1], out_name, use_conv=True, use_bn=True, use_relu=True,
                       num_output=32, kernel_size=3, pad=1, stride=1,
                       dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                       bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                       scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    else:
        NewConvBNLayer(net, net.keys()[-1], out_name, use_conv=True, use_bn=True, use_relu=True,
                       num_output=32, kernel_size=3, pad=2, stride=1,
                       dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                       bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                       scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    block_name = 'concat_{}'.format(out_layer[4:])  # for example, 'conv3_1' to '3_1'
    net[block_name] = L.Concat(net[from_layer], net[net.keys()[-1]])


def DenseNet121Body(net, from_layer, use_pool5=True, use_dilation_conv4=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = ''
    bn_postfix = '/bn'
    scale_prefix = ''
    scale_postfix = '/scale'
    ConvBNLayer(net, from_layer, 'conv1', use_conv=True, use_bn=True, use_relu=True,
                num_output=64, kernel_size=7, pad=3, stride=2,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1, ceil_mode=False)

    # dense block2
    for i in xrange(1, 7):
        conv_name = 'conv2_{}'.format(i)
        DenseBlock(net, net.keys()[-1], conv_name, **bn_param)

    # transition block2
    NewConvBNLayer(net, net.keys()[-1], 'conv2_blk', use_conv=True, use_bn=True, use_relu=True,
                   num_output=128, kernel_size=1, pad=0, stride=1,
                   conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                   bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                   scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool2 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.AVE, kernel_size=2, stride=2, pad=0)

    # dense block3
    for i in xrange(1, 13):
        conv_name = 'conv3_{}'.format(i)
        DenseBlock(net, net.keys()[-1], conv_name, **bn_param)


    # transition block3
    NewConvBNLayer(net, net.keys()[-1], 'conv3_blk', use_conv=True, use_bn=True, use_relu=True,
                   num_output=256, kernel_size=1, pad=0, stride=1,
                   conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                   bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                   scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool3 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.AVE, kernel_size=2, stride=2, pad=0)

    # dense block4
    for i in xrange(1, 25):
        conv_name = 'conv4_{}'.format(i)
        DenseBlock(net, net.keys()[-1], conv_name, **bn_param)

    # transition block4
    NewConvBNLayer(net, net.keys()[-1], 'conv4_blk', use_conv=True, use_bn=True, use_relu=True,
                   num_output=512, kernel_size=1, pad=0, stride=1,
                   conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                   bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                   scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    stride = 2
    dilation = 1
    kernel_size = 2
    if use_dilation_conv4:
        stride = 1
        dilation = 2
        kernel_size = 1

    net.pool4 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.AVE, kernel_size=kernel_size, stride=stride, pad=0)

    # dense block5
    for i in xrange(1, 17):
        conv_name = 'conv5_{}'.format(i)
        DenseBlock(net, net.keys()[-1], conv_name, dilation=dilation, **bn_param)

    # transition block5
    NewConvBNLayer(net, net.keys()[-1], 'conv5_blk', use_conv=False, use_bn=True, use_relu=True,
                   num_output=512, kernel_size=1, pad=0, stride=1,
                   dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                   bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                   scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    if use_pool5:
        net.pool5 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.AVE, global_pooling=True)

    return net
