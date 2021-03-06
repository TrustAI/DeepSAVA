######For one video each a time 
import argparse
import os
import sys
import shutil
import models
import numpy as np
import tensorflow as tf
import scipy.misc
from data import DataSet
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import cv2
import numpy as np
from PIL import Image
from ssim import SSIM
import time
from stn import spatial_transformer_network as transformer
_errstr = "Mode is unknown or incompatible with input array shape."
#import tensorflow_probability as tfp
import stadv
from GA_mask import gen_a,my_sa,sa_tsp,ba_op,ba_op_4
from matplotlib.ticker import FormatStrFormatter
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

def calc_gradients(
        test_file,
        data_set_name,
        model_name,
        output_file_dir,
        max_iter,
        constraint= 'Iteration',
        budget = 0.1,
        learning_rate=0.0001,
        targets=None,
        weight_loss2=1,
        data_spec=None,
        batch_size=1,
        total_len=40,
        seq_len = 10):

    """Compute the gradients for the given network and images."""
    spec = data_spec
    if data_set_name =='UCF101':
        class_no =101
    else:
        class_no = 51
    #initial_T =  np.array([[0.1255,0.5642,20],[1.0041,0,10],[0,0,1]])
    modifier = tf.Variable(0.01*np.ones((1, seq_len, spec.crop_size,spec.crop_size,spec.channels),dtype=np.float32))
    blur_para = tf.Variable(0.01*np.ones((1, seq_len, spec.crop_size,spec.crop_size,spec.channels),dtype=np.float32))
    # identity transform
   
    input_image = tf.placeholder(tf.float32, (batch_size, total_len, spec.crop_size, spec.crop_size, spec.channels))
    input_label = tf.placeholder(tf.int32, (batch_size))
    image_to_rotate = tf.placeholder(shape=(1,spec.crop_size, spec.crop_size, spec.channels), dtype=tf.float32)
    angle_to_rotate = tf.placeholder(shape=(6), dtype=tf.float32)
    
    #theta = tf.placeholder(tf.float32,shape=((seq_len)))
    theta = tf.placeholder(shape=(seq_len), dtype=tf.float32)
    flows_var = tf.placeholder(tf.float32,shape=((1,2, spec.crop_size,spec.crop_size)))
    
    rotate_result = stadv.layers.flow_st( transformer(image_to_rotate,angle_to_rotate), flows_var, 'NHWC')
   
    flows = tf.Variable(np.zeros((seq_len,2, spec.crop_size,spec.crop_size),dtype=np.float32))
    tau = tf.placeholder_with_default(
        tf.constant(0., dtype=tf.float32),
        shape=[], name='tau'
    )
    indicator = tf.placeholder(tf.float32,shape=(seq_len))
    for ll in range(seq_len):
        #if indicator[ll] == 1:
            #the = theta[ll]*indicator[ll]
            #angle = [[tf.cos(the),-tf.sin(the),0.1*indicator[ll]],[tf.sin(the),tf.cos(the),0.1*indicator[ll]]]
        
            #rotate_img = transformer(tf.expand_dims(input_image[0,ll,:,:,:],0),angle)
        #else:
            #rotate_img = tf.expand_dims(input_image[0,ll,:,:,:],0)
        #perturbed_images = tf.minimum(tf.maximum(stadv.layers.flow_st( transformer(tf.expand_dims(input_image[0,ll,:,:,:],0)*255.0,angle), flows[ll]*indicator[ll], 'NHWC'), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        perturbed_images = tf.minimum(tf.maximum(stadv.layers.flow_st( tf.expand_dims(input_image[0,ll,:,:,:],0)*255.0, flows[ll]*indicator[ll], 'NHWC'), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        mask_temp = tf.minimum(tf.maximum(modifier[0,ll,:,:,:]*indicator[ll]+perturbed_images[0]*255.0, -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        mask_temp = tf.expand_dims(mask_temp , 0)
        if ll==0:
            true_image = mask_temp
        else:
            #mask_temp = input_image[0,ll+1,:,:,:]
            #mask_temp = tf.expand_dims(mask_temp,0)
            true_image = tf.concat([true_image, mask_temp],0)
    if seq_len < total_len:
        true_image  = tf.concat([true_image, input_image[0,seq_len:total_len,:,:,:]],0)
    true_image = tf.expand_dims(true_image, 0)
    
    for kk in range(batch_size-1):
        #true_image_temp = tf.minimum(tf.maximum(modifier[0,0,:,:,:]+transformer(tf.expand_dims(input_image[kk+1,0,:,:,:]*255.0,0),angle[0]), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        true_image_temp = tf.minimum(tf.maximum(modifier[0,0,:,:,:]+input_image[kk+1,0,:,:,:]*255.0, -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
        true_image_temp = tf.expand_dims(true_image_temp, 0)
        for ll in range(seq_len-1):
            if indicator[ll+1] == 1:
               mask_temp = tf.minimum(tf.maximum(modifier[0,ll+1,:,:,:]+transformer(tf.expand_dims([input_image[kk+1,ll+1,:,:,:]*255.0],0),angle[0]), -spec.mean+spec.rescale[0]), -spec.mean+spec.rescale[1])/255.0
            else:
               mask_temp = input_image[kk+1,ll+1,:,:,:]
               mask_temp = tf.expand_dims(mask_temp,0)
            true_image_temp = tf.concat([true_image_temp, mask_temp],0)
        true_image_temp = tf.expand_dims(true_image_temp, 0)

        true_image = tf.concat([true_image, true_image_temp],0)
    
    loss2_l12 = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(true_image-input_image), axis=[0, 2, 3, 4])))
    
    loss2 =  tf.reduce_sum(1-tf.image.ssim_multiscale(true_image, input_image, max_val=1.0))
    #loss2 = 1.0 - tf.reduce_mean(SSIM(true_image).cw_ssim_value(input_image))
    
    norm_frame = tf.reduce_mean(tf.abs(modifier), axis=[2,3,4])
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
   
    probs, variable_set, pre_label = models.get_model(sess, true_image, model_name,data_set_name, False)
    true_label_prob = tf.reduce_sum(probs*tf.one_hot(input_label,class_no),[1])
    if model_name == 'i3d_inception':
          true_label_prob = tf.reduce_mean(probs*tf.one_hot(input_label,class_no))
    if targets is None:
        #loss1 = tf.maximum(0.0,true_label_prob)
        loss1 = -tf.log(1 - true_label_prob + 1e-6)
    else:
        loss1 = -tf.log(true_label_prob + 1e-6)
    loss1 = tf.reduce_sum(loss1)
    
    
    loss = loss1 + weight_loss2 * (loss2 +loss2_l12) + tf.reduce_mean(tf.abs(flows))
   
    
    grad_op = tf.gradients(loss,theta)
  
    optimizer = tf.train.AdamOptimizer(learning_rate)
    print('optimizer.minimize....')
    
    train = optimizer.minimize(loss, var_list=[modifier,flows])
    Train = optimizer.minimize(loss1, var_list=[modifier,flows])
    # initiallize all uninitialized varibales
    init_varibale_list = set(tf.all_variables()) - variable_set
    
    sess.run(tf.initialize_variables(init_varibale_list))

    data = DataSet(data_set=data_set_name,test_list=test_file, seq_length=seq_len,image_shape=(spec.crop_size, spec.crop_size, spec.channels))
    print('data loaded')
    all_names = []
    all_images = []
    all_labels = []
    output_names = []
    def_len = seq_len
    for video in data.test_data:
        frames,f_name = data.get_frames_for_sample(data_set_name,video)
        if len(frames) < def_len:
           continue
        frames = data.rescale_list(frames, def_len)
        frames_data = data.build_image_sequence(frames)
        all_images.append(frames_data)
        label, hot_labels = data.get_class_one_hot(video[1])
        all_labels.append(label)
        all_names.append(f_name)
        output_names.append(frames)
    total = len(all_names)
    all_indices = range(total)
    num_batch = int(total/batch_size)
    f = open("rotate_ssim_hcm.txt", "a+")
    print('process data length:', num_batch,file=f)

    correct_ori = 0
    correct_noi = 0
    tot_image = 0
    adv = 0
    sess.run(tf.initialize_variables(init_varibale_list))
    for ii in range(num_batch):
        images = all_images[ii*batch_size : (ii+1)*batch_size]
        names = all_names[ii*batch_size : (ii+1)*batch_size]
        labels = all_labels[ii*batch_size : (ii+1)*batch_size]
        indices = all_indices[ii*batch_size : (ii+1)*batch_size]
        output_name = output_names[ii*batch_size : (ii+1)*batch_size]
        print('------------------prediction for clean video-------------------')
        print('---video-level prediction---')
        
        for xx in range(len(indices)):
            print(names[xx],'label:', labels[xx], 'indice:',indices[xx], 'size:', len(images[xx]), len(images[xx][0]), len(images[xx][0][0]), len(images[xx][0][0][0]))
        
        sess.run(tf.initialize_variables(init_varibale_list))
        
        if targets is not None:
            labels = [targets[e] for e in names]
        
        
        #feed_dict = {input_image: images[0:seq_len], input_label: labels,tau: 0.05,flows:null_flows,indicator:indicator_ini,theta:np.zeros((seq_len))}
        indicator_ini =  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        feed_dict = {input_image: images[0:total_len], input_label: labels,tau: 0.05,indicator:indicator_ini,theta:np.zeros((seq_len))}
        var_loss, true_prob, var_loss1, var_loss2, var_l12loss,var_pre= sess.run((loss, true_label_prob, loss1, loss2,loss2_l12, pre_label), feed_dict=feed_dict)
        
        correct_pre = correct_ori
        for xx in range(len(indices)):
           if labels[xx] == var_pre[xx]:
              correct_ori += 1

        tot_image += 1
        print('Start!')
        min_loss = var_loss
        last_min = -1
        print('---frame-wise prediction---')
        #print('node_label:', var_node, 'label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib', true_prob)
        print('label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib', true_prob,'var_l12loss',var_l12loss)
        # record numer of iteration
        tot_iter = 0
        
        if correct_pre == correct_ori:#if model predict is wrong
           ii += 1
           continue
        if true_prob ==1.0:
            ii +=1
            correct_noi +=1
            continue
       
        print('------------------prediction for adversarial video-------------------')
        Test_mode = True
        ge_time =time.time()
        theta_in = np.ones((seq_len))*0.5
        
        index = ba_op(train,init_varibale_list,true_label_prob,seq_len,indicator,f,feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05,theta : theta_in},sess=sess)
        
        print(index)
        mask = np.zeros((seq_len))
        mask[index] =1
        ######## without BO selection ##############
        ###### select four frames by BO###########
        '''
        ind1,ind2,ind3,ind4 = ba_op_4(train,init_varibale_list,true_label_prob,seq_len,indicator,f,feed_dict={input_image: images[0:seq_len], input_label: labels, tau: 0.05},sess=sess)
        mask = np.zeros((seq_len))
        for i in [ind1,ind2,ind3,ind4]:
            mask[i] = 1
        '''
        print(mask)
        sess.run(tf.initialize_variables(init_varibale_list))
        
        
        start_loss = var_loss1
        if ii < 400:
            Test_mode = False
            for cur_iter in range(max_iter):
                start_time = time.time()
                tot_iter += 1
                    
                    
                sess.run(train, feed_dict=feed_dict)
                var_loss,true_prob,var_loss1, var_loss2, var_l12loss,var_pre= sess.run((loss, true_label_prob, loss1, loss2,loss2_l12, pre_label), feed_dict=feed_dict)
                print('iter:', cur_iter, 'total loss:', var_loss, 'label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib:', true_prob,'var_l12loss',var_l12loss)
                
                print('time',time.time()-start_time)
                break_condition = False
                if constraint == 'ssim':
                    lo = var_loss2
                elif constraint == 'lp':
                    lo = var_losslp
                else:
                    lo = 0
                if lo > budget:
                     break_condition = True
                
                if var_loss < min_loss:
                    if np.absolute(var_loss-min_loss) < 0.00001:
                        break_condition = True
                        print(last_min)
                        min_loss = var_loss
                        last_min = cur_iter
                
                
                if cur_iter + 1 == max_iter or break_condition:
                    print('iter:', cur_iter,  'label loss:', var_loss1, 'content loss:', var_loss2, 'prediction:', var_pre, 'probib:', true_prob,'var_l12loss',var_l12loss)
                    var_diff, flows_var, var_probs, noise_norm = sess.run((modifier, flows, probs, norm_frame), feed_dict=feed_dict)
                    
                    #for pp in range(seq_len):
                     #print the map value for each frame
                        #print(noise_norm[0][pp])
                    for i in range(len(indices)):
                        top1 = var_probs[i].argmax()
                        if labels[i] == top1:
                            correct_noi += 1
                    np.save('flow_st_only.npy',flows_var)
                    np.save('modifier_st_only.npy',var_diff)
                    break
                
                    
                
                    
            print('saved modifier paramters.', ii,'spend time',time.time()-start_time)
 
        
       
        ###### save images #########
        true_im= sess.run(true_image, feed_dict=feed_dict)
        
     
        for ll in range(len(indices)):
            for kk in range(def_len):
                if kk < seq_len:
                    #if indicator[kk] == 1:
                    
                    attack_image = true_im[ll][kk]
                        #np.reshape(angle_var,(6))
                    
                    attack_img = np.clip(attack_image*255.0+data_spec.mean,data_spec.rescale[0],data_spec.rescale[1])
                        
                    #else:
                        #attack_img = np.clip(images[ll][kk]*255.0+var_diff[0][kk]+data_spec.mean,data_spec.rescale[0],data_spec.rescale[1])
                   
                    diff = np.clip(np.absolute(var_diff[0][kk])*255.0, data_spec.rescale[0],data_spec.rescale[1])
                else:
                   attack_img = np.clip(images[ll][kk]*255.0+data_spec.mean,data_spec.rescale[0],data_spec.rescale[1])
                   diff = np.zeros((spec.crop_size,spec.crop_size,spec.channels))
                im_diff = toimage(arr=diff, cmin=data_spec.rescale[0], cmax=data_spec.rescale[1])
                im = toimage(arr=attack_img, cmin=data_spec.rescale[0], cmax=data_spec.rescale[1])
                new_name = output_name[ll][kk].split('/')
                
                adv_dir = output_file_dir+'/adversarial_100/'
                dif_dir = output_file_dir+'/noise_100/'
                if not os.path.exists(adv_dir):
                   os.mkdir(adv_dir)
                   os.mkdir(dif_dir)

                tmp_dir = adv_dir+new_name[-2]
                tmp1_dir = dif_dir+new_name[-2]
                if not os.path.exists(tmp_dir):
                   os.mkdir(tmp_dir)
                   os.mkdir(tmp1_dir)
               
                new_name = new_name[-1] + '.png'
                im.save(tmp_dir + '/' +new_name)
                im_diff.save(tmp1_dir + '/' +new_name)
  
        
        #print('saved adversarial frames.', ii,file=f)
        
        #print('correct_ori:', correct_ori, 'correct_noi:', correct_noi,'adv_examples',adv,file=f)
        print('correct_ori:', correct_ori, 'correct_noi:', correct_noi,'adv_examples',adv)
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Use Adam optimizer to generate adversarial examples.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory of dataset.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory of output image file.')
    parser.add_argument('--dataset', type=str, required=True,choices=['UCF101','HMDB51'],
                help='dataset to be evaluated.')
    parser.add_argument('--model', type=str, required=True,choices=['GoogleNet','Inception2','i3d_inception','I3V','LSTM','i3d','resnet','tsn','c3d'],
                help='Models to be evaluated.')
    parser.add_argument('--num_images', type=int, default=sys.maxsize,
                        help='Max number of images to be evaluated.')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Evaluate a specific list of file in dataset.')
    parser.add_argument('--num_iter', type=int, default=100,
                        help='Number of iterations to generate attack.')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save .npy file when each save_freq iterations.')
    parser.add_argument('--learning_rate', type=float, default=0.001 * 255,
                        help='Learning rate of each iteration.')
    parser.add_argument('--target', type=str, default=None,
                        help='Target list of dataset.')
    parser.add_argument('--weight_loss2', type=float, default=1.0,
                        help='Weight of distance penalty.')
    parser.add_argument('--not_crop', dest='use_crop', action='store_false',
                        help='Not use crop in image producer.')
    parser.add_argument('--constraint', type=str, required=True,choices=['iteration','ssim','lp'],
                        help='the type of consraint to stop optimizations.')
    parser.add_argument('--budget', type=float, default=0.1,
                        help='the budget of contraint.')

    parser.set_defaults(use_crop=True)
    args = parser.parse_args()
    print(args.file_list)
    assert args.num_iter % args.save_freq == 0

    data_spec = models.get_data_spec(model_name=args.model)
    args.learning_rate = args.learning_rate / 255.0 * (data_spec.rescale[1] - data_spec.rescale[0])
    seq_len = 40
    total_len = 40
    batch_size = 1
    targets = None
    if args.target is not None:
        targets = {}
        with open(args.target, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                targets[key] = int(value)
                
    calc_gradients(
        args.file_list,
        args.dataset,
        args.model,
        args.output_dir,
        args.num_iter,
        args.constraint,
        args.budget,
        args.learning_rate,
        targets,
        args.weight_loss2,
        data_spec,
        batch_size,
        total_len,
        seq_len)
    
if __name__ == '__main__':
    main()

