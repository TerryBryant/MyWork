from __future__ import print_function
caffe_root = '/home/terry/software/caffe-ssd-gpu'   # should set to your caffe root path
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

from caffe import layers as L
from caffe import params as P


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
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def invert_residual_block(net, from_layer, out_layer, out_expand, out_dwise, out_linear, stride,
                          has_shortcut=False, dilation=1, **bn_param):
    conv_prefix = '{}/'.format(out_layer)
    conv_postfix = ''
    bn_postfix = ''
    scale_postfix = ''
    use_scale = True


    # expand
    expand_name = conv_prefix + 'expand'
    ConvBNLayer(net, from_layer, '', use_bn=True, use_relu=True,
                num_output=out_expand, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=expand_name, conv_postfix=conv_postfix,
                bn_prefix=expand_name + '/bn', bn_postfix=bn_postfix,
                scale_prefix=expand_name + '/scale', scale_postfix=scale_postfix, **bn_param)
    out_name = '{}{}'.format(expand_name, '_relu')

    # depthwise, caffe dosen't support depthwise convolution, which is a special kind of group convolution
    # Here I recommend a third-party implement: https://github.com/yonghenglh6/DepthwiseConvolution
    expand_name = conv_prefix + 'dwise'
    if dilation == 1:
        ConvBNLayer(net, out_name, '', use_bn=True, use_relu=True,
                    num_output=out_dwise, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
                    use_group=True, conv_prefix=expand_name, conv_postfix=conv_postfix,
                    bn_prefix=expand_name + '/bn', bn_postfix=bn_postfix,
                    scale_prefix=expand_name + '/scale', scale_postfix=scale_postfix, **bn_param)
    else:
        pad = int((3 + (dilation - 1) * 2) - 1) / 2
        ConvBNLayer(net, out_name, '', use_bn=True, use_relu=True,
                    num_output=out_dwise, kernel_size=3, pad=pad, stride=stride, use_scale=use_scale,
                    use_group=True, dilation=dilation, conv_prefix=expand_name, conv_postfix=conv_postfix,
                    bn_prefix=expand_name + '/bn', bn_postfix=bn_postfix,
                    scale_prefix=expand_name + '/scale', scale_postfix=scale_postfix, **bn_param)
    out_name = '{}{}'.format(expand_name, '_relu')

    # linear
    expand_name = conv_prefix + 'linear'
    ConvBNLayer(net, out_name, '', use_bn=True, use_relu=False,
                num_output=out_linear, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=expand_name, conv_postfix=conv_postfix,
                bn_prefix=expand_name + '/bn', bn_postfix=bn_postfix,
                scale_prefix=expand_name + '/scale', scale_postfix=scale_postfix, **bn_param)
    out_name = '{}{}'.format(expand_name, '/scale')

    if has_shortcut:
        res_name = 'block_{}'.format(out_layer[4:])   # for example, 'conv3_1' to '3_1'
        net[res_name] = L.Eltwise(net[from_layer], net[out_name])


def MobilenetV2Body(net, from_layer, use_pool6=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = 'bn_'
    bn_postfix = ''
    scale_prefix = 'scale_'
    scale_postfix = ''
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
                num_output=32, kernel_size=3, pad=1, stride=2,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    invert_residual_block(net, net.keys()[-1], 'conv2_1', 32, 32, 16, 1, False, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv2_2', 96, 96, 24, 2, False, **bn_param)

    invert_residual_block(net, net.keys()[-1], 'conv3_1', 144, 144, 24, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv3_2', 144, 144, 32, 2, False, **bn_param)

    invert_residual_block(net, net.keys()[-1], 'conv4_1', 192, 192, 32, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv4_2', 192, 192, 32, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv4_3', 192, 192, 64, 1, False, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv4_4', 384, 384, 64, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv4_5', 384, 384, 64, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv4_6', 384, 384, 64, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv4_7', 384, 384, 96, 2, False, **bn_param)

    invert_residual_block(net, net.keys()[-1], 'conv5_1', 576, 576, 96, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv5_2', 576, 576, 96, 1, True, **bn_param)

    if use_dilation_conv5:
        dilation = 2
    invert_residual_block(net, net.keys()[-1], 'conv5_3', 576, 576, 160, 2, False, dilation=dilation, **bn_param)

    invert_residual_block(net, net.keys()[-1], 'conv6_1', 960, 960, 160, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv6_2', 960, 960, 160, 1, True, **bn_param)
    invert_residual_block(net, net.keys()[-1], 'conv6_3', 960, 960, 320, 1, False, **bn_param)

    ConvBNLayer(net, net.keys()[-1], '', use_bn=True, use_relu=True,
                num_output=1280, kernel_size=1, pad=0, stride=1,
                conv_prefix='conv6_4', conv_postfix='',
                bn_prefix='conv6_4' + '/bn', bn_postfix='',
                scale_prefix='conv6_4' + '/scale', scale_postfix='', **bn_param)

    if use_pool6:
        net.pool6 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.AVE, global_pooling=True)

    return net
