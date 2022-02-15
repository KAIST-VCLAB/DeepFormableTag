"""
Inpaint codes of Deepfill are copied from Deepfill repository: https://github.com/JiahuiYu/generative_inpainting
and modified by Mustafa B. Yaldiz (VCLAB, KAIST). All Rights Reserved.
"""
import pathlib
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.client import device_lib
# from tensorflow.python.compiler.tensorrt import trt_convert

def resize(x, scale=2, to_shape=None, align_corners=True,
           func=tf.image.resize_bilinear, name='resize'):
    
    xs = tf.cast(tf.shape(x), tf.float32)
    new_xs = [tf.cast(xs[1]*scale, tf.int32),
                tf.cast(xs[2]*scale, tf.int32)]

    with tf.variable_scope(name):
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.nn.elu, training=True):
    """Define conv for generator.
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)
    if cnum == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3)
    x = activation(x)
    y = tf.nn.sigmoid(y)
    x = x * y
    return x

@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.
    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.
    Args:
        mask: Original mask.
        x: To shape of x.
    Returns:
        tf.Tensor: resized mask
    """
    mask_resize = resize(
        mask, to_shape=tf.shape(x)[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.
    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.
    Returns:
        tf.Tensor: output
    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    out_size = tf.cast(tf.shape(b)[1:3] / rate, tf.int32)
    b = tf.image.resize_nearest_neighbor(f, size=out_size)
    # b = resize(b, to_shape=[tf.cast(raw_int_bs[1]/rate, tf.int32), tf.cast(raw_int_bs[2]/rate, tf.int32)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_bilinear)
    return y, flow


def build_inpaint_net(x, mask, reuse=False, training=True,padding='SAME', name='inpaint_net'):
    """Inpaint network.
    Args:
        x: incomplete image, [-1, 1]
        mask: mask region {0, 1}
    Returns:
        [-1, 1] as predicted image
    """
    xin = x
    offset_flow = None
    ones_x = tf.ones_like(x)[:, :, :, 0:1]
    x = tf.concat([x, ones_x, ones_x*mask], axis=3)

    # two stage network
    cnum = 48
    with tf.variable_scope(name, reuse=reuse), \
            arg_scope([gen_conv, gen_deconv],
                        training=training, padding=padding):
        # stage1
        x = gen_conv(x, cnum, 5, 1, name='conv1')
        x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')
        x = gen_conv(x, 2*cnum, 3, 1, name='conv3')
        x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')
        x = gen_conv(x, 4*cnum, 3, 1, name='conv5')
        x = gen_conv(x, 4*cnum, 3, 1, name='conv6')
        # This requires image shape
        mask_s = resize_mask_like(mask, x)
        x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')
        x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')
        x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')
        x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')
        x = gen_conv(x, 4*cnum, 3, 1, name='conv11')
        x = gen_conv(x, 4*cnum, 3, 1, name='conv12')
        x = gen_deconv(x, 2*cnum, name='conv13_upsample')
        x = gen_conv(x, 2*cnum, 3, 1, name='conv14')
        x = gen_deconv(x, cnum, name='conv15_upsample')
        x = gen_conv(x, cnum//2, 3, 1, name='conv16')
        x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
        x = tf.nn.tanh(x)
        x_stage1 = x

        # stage2, paste result as input
        x = x*mask + xin[:, :, :, 0:3]*(1.-mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        # conv branch
        # xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)
        xnow = x
        x = gen_conv(xnow, cnum, 5, 1, name='xconv1')
        x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')
        x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')
        x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')
        x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')
        x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')
        x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')
        x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')
        x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')
        x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')
        x_hallu = x
        # attention branch
        x = gen_conv(xnow, cnum, 5, 1, name='pmconv1')
        x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample')
        x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')
        x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')
        x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')
        x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',
                            activation=tf.nn.relu)
        x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
        x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')
        x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')
        pm = x
        x = tf.concat([x_hallu, pm], axis=3)

        x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')
        x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')
        x = gen_deconv(x, 2*cnum, name='allconv13_upsample')
        x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')
        x = gen_deconv(x, cnum, name='allconv15_upsample')
        x = gen_conv(x, cnum//2, 3, 1, name='allconv16')
        x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
        x = tf.nn.tanh(x)
        x_stage2 = x
    return x_stage1, x_stage2, offset_flow

# TODO: Check whether network output is different than actual image
def build_graph(batch_raw, masks_raw, reuse=False):
    # batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
    # masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)
    masks = masks_raw[0:1,:,:,0:1]

    batch_pos = batch_raw / 127.5 - 1.
    batch_incomplete = batch_pos * (1. - masks)
    xin = batch_incomplete
    # inpaint
    x1, x2, flow = build_inpaint_net(xin, masks, reuse=reuse, training=False)
    batch_predict = x2
    # apply mask and reconstruct
    batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
    output = (batch_complete + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    return output

def get_gpu_list():
    return [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

def init_inpaint_network(sess, gpu_list=None):    
    checkpoint_dir = str(pathlib.Path(__file__).parent.absolute() / 'inpaint_weights')
    inpaint_inputs = []
    inpaint_outputs = []

    if gpu_list is None:
        gpu_list = get_gpu_list()

    for i, device in enumerate(gpu_list):
        with tf.variable_scope('gpu'+str(i), reuse=False), tf.device(device):
            batch_raw = tf.placeholder(tf.float32, shape=(1,None,None,3))
            masks_raw = tf.placeholder(tf.float32, shape=(1,None,None,3))
            input_op = (batch_raw, masks_raw)
            output = build_graph(batch_raw, masks_raw, reuse=False)
            inpaint_inputs.append(input_op)
            inpaint_outputs.append(output)
    
    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name[5:]
        from_name = vname
        var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded to multi_gpus')
    return inpaint_inputs, inpaint_outputs