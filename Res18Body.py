import caffe
from caffe.model_libs import *

def Res18Body(net, from_layer, block_name, out2a, out2b, stride, use_branch1, dilation=1, **bn_param):
    conv_prefix = "res{}_".format(block_name)
    conv_postfix = ""
    bn_prefix = "bn{}_".format(block_name)
    bn_postfix = ""
    scale_prefix = "scale{}_".format(block_name)
    scale_postfix = ""
    use_scale = True
    if use_branch1:
        branch_name = "branch1"
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
            num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
            conv_prefix=conv_prefix, conv_postfix=conv_postfix,
            bn_prefix=bn_prefix, bn_postfix=bn_postfix,
            scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
        branch1 = "{}{}".format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = "branch2a"
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
        num_output=out2a, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param) 
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = "branch2b"
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
        num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
        conv_prefix=conv_prefix, conv_postfix=conv_postfix,
        bn_prefix=bn_prefix, bn_postfix=bn_postfix,
        scale_prefix=scale_prefix, scale_postfix=scale_postfix, **bn_param)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = "res{}".format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)

def MyResBody(net, from_layer):
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
            num_output=64, kernel_size=7, pad=3, stride=2,
            conv_prefix="", conv_postfix="",
            bn_prefix="bn_", bn_postfix="",
            scale_prefix="scale_", scale_postfix="")

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    Res18Body(net, "pool1", "2a", 64, 64, 1, True)
    Res18Body(net, "res2a", "2b", 64, 64, 1, False)
    Res18Body(net, "res2b", "3a", 128, 128, 2, True)
    Res18Body(net, "res3a", "3b", 128, 128, 1, False)
    Res18Body(net, "res3b", "4a", 256, 256, 2, True)
    Res18Body(net, "res4a", "4b", 256, 256, 1, False)
    Res18Body(net, "res4b", "5a", 512, 512, 2, True)
    Res18Body(net, "res5a", "5b", 512, 512, 1, False)
    return net











