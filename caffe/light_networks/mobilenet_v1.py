from __future__ import print_function
caffe_root = '/home/terry/software/caffe-ssd-gpu'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')


import caffe
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
        relu_name = 'relu{}'.format(conv_name.strip('conv'))
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def depthwise_block(net, from_layer, out_layer, out_dwise, out_sep, stride, dilation=1, **bn_param):
    conv_prefix = ''
    conv_postfix = out_layer
    bn_prefix = ''
    bn_postfix = out_layer
    scale_prefix = ''
    scale_postfix = out_layer
    use_scale = True

    # depthwise convolution
    ConvBNLayer(net, from_layer, '', use_bn=True, use_relu=True,
                num_output=out_dwise, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
                use_group=True, conv_prefix=conv_prefix, conv_postfix=conv_postfix + '/dw',
                bn_prefix=bn_prefix, bn_postfix=bn_postfix + '/dw/bn',
                scale_prefix=scale_prefix, scale_postfix=scale_postfix + '/dw/scale', **bn_param)


    # seperate
    if dilation == 1:
        ConvBNLayer(net, net.keys()[-1], '', use_bn=True, use_relu=True,
                    num_output=out_sep, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                    dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix + '/sep',
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix + '/sep/bn',
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix + '/sep/scale', **bn_param)
    else:
        pad = int((3 + (dilation - 1) * 2) - 1) / 2
        ConvBNLayer(net, net.keys()[-1], '', use_bn=True, use_relu=True,
                    num_output=out_sep, kernel_size=1, pad=pad, stride=1, use_scale=use_scale,
                    dilation=dilation, conv_prefix=conv_prefix, conv_postfix=conv_postfix + '/sep',
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix + '/sep/bn',
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix + '/sep/scale', **bn_param)


def MobilenetV1Body(net, from_layer, use_pool6=True, use_dilation_conv5=False, **bn_param):
    conv_prefix = ''
    conv_postfix = ''
    bn_prefix = ''
    bn_postfix = '/bn'
    scale_prefix = ''
    scale_postfix = '/scale'
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
                num_output=32, kernel_size=3, pad=1, stride=2,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)

    depthwise_block(net, net.keys()[-1], 'conv2_1', 32, 64, stride=1, **bn_param)
    depthwise_block(net, net.keys()[-1], 'conv2_2', 64, 128, stride=2, **bn_param)

    depthwise_block(net, net.keys()[-1], 'conv3_1', 128, 128, stride=1, **bn_param)
    depthwise_block(net, net.keys()[-1], 'conv3_2', 128, 256, stride=2, **bn_param)

    depthwise_block(net, net.keys()[-1], 'conv4_1', 256, 256, stride=1, **bn_param)
    depthwise_block(net, net.keys()[-1], 'conv4_2', 256, 512, stride=2, **bn_param)

    for i in xrange(1, 6):
        conv_name = 'conv5_{}'.format(i)
        depthwise_block(net, net.keys()[-1], conv_name, 512, 512, stride=1, **bn_param)

    stride = 2
    dilation = 1
    if use_dilation_conv5:
        stride = 1
        dilation = 2

    depthwise_block(net, net.keys()[-1], 'conv5_6', 512, 1024, stride=stride, dilation=dilation, **bn_param)

    depthwise_block(net, net.keys()[-1], 'conv6', 1024, 1024, stride=1, dilation=dilation, **bn_param)

    if use_pool6:
        net.pool6 = L.Pooling(net[net.keys()[-1]], pool=P.Pooling.AVE, global_pooling=True)




net = caffe.NetSpec()
net.data = L.Data(batch_size=1, backend=P.Data.LMDB, source='train_lmdb',
                             transform_param=dict(scale=1./255, crop_size=224)
                             )
MobilenetV1Body(net, from_layer='data', use_pool6=True, use_dilation_conv5=True)

kwargs={
    'param': [
        dict(lr_mult=1, decay_mult=1),
        dict(lr_mult=2, decay_mult=0)],
    'weight_filler': dict(type='msra'),
    'bias_filler': dict(type='constant', value=0)
}
net.fc7 = L.Convolution(net[net.keys()[-1]], num_output=1000, kernel_size=1,
                        pad=0, stride=1, **kwargs)
net.prob = L.Softmax(net[net.keys()[-1]])




with open('train_mobilenetv1.prototxt', 'w') as f:
    print('name: "{}_train"'.format('mobilenetv1'), file=f)
    print(net.to_proto(), file=f)

