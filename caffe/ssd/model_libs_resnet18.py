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
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='_bn',
    scale_prefix='', scale_postfix='_scale', bias_prefix='', bias_postfix='_bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
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
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
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
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def ResShallowBody(net, from_layer, block_name, out2a, out2b, stride, use_branch1, dilation=1, **bn_param):
    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    if dilation == 1:
        ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
                    num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    else:
        pad = int((3 + (dilation - 1) * 2) - 1) / 2
        ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
                    num_output=out2b, kernel_size=3, pad=pad, stride=1, use_scale=use_scale,
                    dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)


def ResNet18Body(net, from_layer, use_pool5=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
                num_output=64, kernel_size=7, pad=3, stride=2,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResShallowBody(net, 'pool1', '2a', out2a=64, out2b=64, stride=1, use_branch1=True, **bn_param)
    ResShallowBody(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False, **bn_param)

    ResShallowBody(net, 'res2b', '3a', out2a=128, out2b=128, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res3a'
    for i in xrange(1, 2):
        block_name = '3b{}'.format(i)
        ResShallowBody(net, from_layer, block_name, out2a=128, out2b=128, stride=1, use_branch1=False, **bn_param)
        from_layer = 'res{}'.format(block_name)

    ResShallowBody(net, from_layer, '4a', out2a=256, out2b=256, stride=2, use_branch1=True, **bn_param)

    from_layer = 'res4a'
    for i in xrange(1, 2):
        block_name = '4b{}'.format(i)
        ResShallowBody(net, from_layer, block_name, out2a=256, out2b=256, stride=1, use_branch1=False, **bn_param)
        from_layer = 'res{}'.format(block_name)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
        stride = 1
        dilation = 2

    ResShallowBody(net, from_layer, '5a', out2a=512, out2b=512, stride=stride, use_branch1=True, dilation=dilation, **bn_param)
    ResShallowBody(net, 'res5a', '5b', out2a=512, out2b=512, stride=1, use_branch1=False, dilation=dilation, **bn_param)

    if use_pool5:
        net.pool5 = L.Pooling(net.res5b, pool=P.Pooling.AVE, global_pooling=True)

    return net


