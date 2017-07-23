from __future__ import division

import sys
import os
import argparse
import pdb
import numpy as np
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt

from LSTM_model import LSTM_model
from RMI_model import RMI_model
from pydensecrf import densecrf

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools


def train(modelname, max_iter, snapshot, dataset, weights, setname, mu):
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    tfmodel_folder = './' + dataset + '/tfmodel/'
    snapshot_file = tfmodel_folder + dataset + '_' + weights + '_' + modelname + '_iter_%d.tfmodel'
    if not os.path.isdir(tfmodel_folder):
        os.makedirs(tfmodel_folder)

    cls_loss_avg = 0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
    decay = 0.99
    vocab_size = 8803 if dataset == 'referit' else 12112

    if modelname == 'LSTM':
        model = LSTM_model(mode='train', vocab_size=vocab_size, weights=weights)
    elif modelname == 'RMI':
        model = RMI_model(mode='train', vocab_size=vocab_size, weights=weights)
    else:
        raise ValueError('Unknown model name %s' % (modelname))

    if weights == 'resnet':
        pretrained_model = './external/TF-resnet/model/ResNet101_init.tfmodel'
        load_var = {var.op.name: var for var in tf.global_variables() if var.op.name.startswith('ResNet')}
    elif weights == 'deeplab':
        pretrained_model = './external/TF-deeplab/model/ResNet101_train.tfmodel'
        load_var = {var.op.name: var for var in tf.global_variables() if var.op.name.startswith('DeepLab/group')}

    snapshot_loader = tf.train.Saver(load_var)
    snapshot_saver = tf.train.Saver(max_to_keep = 1000)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    snapshot_loader.restore(sess, pretrained_model)

    reader = data_reader.DataReader(data_folder, data_prefix)
    for n_iter in range(max_iter):

        batch = reader.read_batch()
        text = batch['text_batch']
        im = batch['im_batch'].astype(np.float32)
        mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

        im = im[:,:,::-1]
        im -= mu

        _, cls_loss_val, lr_val, scores_val, label_val = sess.run([model.train_step, 
            model.cls_loss, 
            model.learning_rate, 
            model.pred, 
            model.target], 
            feed_dict={
                model.words: np.expand_dims(text, axis=0),
                model.im: np.expand_dims(im, axis=0),
                model.target_fine: np.expand_dims(mask, axis=0)
            })
        cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
        print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f' % (n_iter, cls_loss_val, cls_loss_avg, lr_val))

        # Accuracy
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
        avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
        avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
        print('iter = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
              % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
        print('iter = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
              % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

        # Save snapshot
        if (n_iter+1) % snapshot == 0 or (n_iter+1) >= max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter+1))
            print('snapshot saved to ' + snapshot_file % (n_iter+1))

    print('Optimization done.')


def test(modelname, iter, dataset, visualize, weights, setname, dcrf, mu):
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    if visualize:
        save_dir = './' + dataset + '/visualization/' + modelname + '_' + str(iter) + '/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    pretrained_model = './' + dataset + '/tfmodel/' + dataset + '_' + weights + '_' + modelname + '_iter_' + str(iter) + '.tfmodel'
    
    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    if dcrf:
        cum_I_dcrf, cum_U_dcrf = 0, 0
        seg_correct_dcrf = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0.
    H, W = 320, 320
    vocab_size = 8803 if dataset == 'referit' else 12112

    if modelname == 'LSTM':
        model = LSTM_model(H=H, W=W, mode='eval', vocab_size=vocab_size, weights=weights)
    elif modelname == 'RMI':
        model = RMI_model(H=H, W=W, mode='eval', vocab_size=vocab_size, weights=weights)
    else:
        raise ValueError('Unknown model name %s' % (modelname))

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    snapshot_restorer.restore(sess, pretrained_model)
    reader = data_reader.DataReader(data_folder, data_prefix, shuffle=False)

    for n_iter in range(reader.num_batch):

        batch = reader.read_batch()
        text = batch['text_batch']
        im = batch['im_batch']
        mask = batch['mask_batch'].astype(np.float32)

        proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
        proc_im_ = proc_im.astype(np.float32)
        proc_im_ = proc_im_[:,:,::-1]
        proc_im_ -= mu

        scores_val, up_val, sigm_val = sess.run([model.pred, model.up, model.sigm],
            feed_dict={
                model.words: np.expand_dims(text, axis=0),
                model.im: np.expand_dims(proc_im_, axis=0)
            })

        # scores_val = np.squeeze(scores_val)
        # pred_raw = (scores_val >= score_thresh).astype(np.float32)
        up_val = np.squeeze(up_val)
        pred_raw = (up_val >= score_thresh).astype(np.float32)
        predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])
        if dcrf:
            # Dense CRF post-processing
            sigm_val = np.squeeze(sigm_val)
            d = densecrf.DenseCRF2D(W, H, 2)
            U = np.expand_dims(-np.log(sigm_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
            Q = d.inference(5)
            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
            predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])

        if visualize:
            sent = batch['sent_batch'][0]
            visualize_seg(im, predicts, sent)
            if dcrf:
                visualize_seg(im, predicts_dcrf, sent)

        I, U = eval_tools.compute_mask_IU(predicts, mask)
        cum_I += I
        cum_U += U
        msg = 'cumulative IoU = %f' % (cum_I/cum_U)
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (I/U >= eval_seg_iou)
        if dcrf:
            I_dcrf, U_dcrf = eval_tools.compute_mask_IU(predicts_dcrf, mask)
            cum_I_dcrf += I_dcrf
            cum_U_dcrf += U_dcrf
            msg += '\tcumulative IoU (dcrf) = %f' % (cum_I_dcrf/cum_U_dcrf)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct_dcrf[n_eval_iou] += (I_dcrf/U_dcrf >= eval_seg_iou)
        print(msg)
        seg_total += 1

    # Print results
    print('Segmentation evaluation (without DenseCRF):')
    result_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result_str += 'precision@%s = %f\n' % \
            (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou]/seg_total)
    result_str += 'overall IoU = %f\n' % (cum_I/cum_U)
    print(result_str)
    if dcrf:
        print('Segmentation evaluation (with DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                (str(eval_seg_iou_list[n_eval_iou]), seg_correct_dcrf[n_eval_iou]/seg_total)
        result_str += 'overall IoU = %f\n' % (cum_I_dcrf/cum_U_dcrf)
        print(result_str)


def visualize_seg(im, predicts, sent):
    im_seg = im / 2
    im_seg[:, :, 0] += predicts.astype('uint8') * 100
    plt.imshow(im_seg.astype('uint8'))
    plt.title(sent)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type = str, default = '0')
    parser.add_argument('-m', type = str) # 'train' 'test'
    parser.add_argument('-n', type = str) # 'LSTM' 'RMI'
    parser.add_argument('-i', type = int, default = 750000)
    parser.add_argument('-s', type = int, default = 50000)
    parser.add_argument('-d', type = str) # 'Gref' 'unc' 'unc+' 'referit'
    parser.add_argument('-v', default = False, action = 'store_true')
    parser.add_argument('-c', default = False, action = 'store_true') # whether or not apply DenseCRF
    parser.add_argument('-w', type = str) # 'resnet' 'deeplab'
    parser.add_argument('-t', type = str) # 'train' 'trainval' 'val' 'test' 'testA' 'testB'

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    if args.m == 'train':
        train(modelname = args.n, 
              max_iter = args.i, 
              snapshot = args.s, 
              dataset = args.d, 
              weights = args.w,
              setname = args.t,
              mu = mu)
    elif args.m == 'test':
        test(modelname = args.n, 
             iter = args.i, 
             dataset = args.d, 
             visualize = args.v,
             weights = args.w,
             setname = args.t,
             dcrf = args.c,
             mu = mu)