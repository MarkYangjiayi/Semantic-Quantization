# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import csv
sys.path.append('../utils')
import segmentation_dataset
import input_generator
from loss import loss
from quantization import QG
from deeplab_v3 import Deeplab_v3
import scipy.misc as misc

note = "std_quant"
mode = "train"#mode can be train or val
logs_dir="../save/" + note + "/ckpt/"
sums_dir="../save/" + note + "/summary/"
loss_dir="../save/" + note + "/loss/"
vis_dir="../save/" + note + "/vis/"

# Choose the GPU to run on
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class args:
    display = 1
    write_summary = 5000
    try_save = 1000
    weight_decay = 1e-5
    batch_norm_decay = 0.95
    batchSize = 8
    testBatchSize = 1# This should not change to ensure correct validation mIoU
    crop_size = [512,512]#size to crop, automatically padded
    lr_schedule = [1, 0.0625, 3000, 0.03125, 12000, 0.015625, 20000, 0.0078125]#learning rate for PASCAL VOC 2012 dataset
    #lr_schedule = [1, 0.0625, 50000, 0.03125, 100000, 0.015625, 150000, 0.0078125]#learning rate for ADE20K dataset
    if mode == "train":
        is_training = True
    else:
        is_training = False

# class dataset:
#     name = "ade20k"
#     train = 20210       #number of training images
#     trainaug = 0
#     trainval = 0
#     val = 2000
#     classes = 151           #classes including ignore_label
#     ignore_label = 0        #label that does not participate training and inference
#     train_steps = 150000
#     data_dir = "../../dataSet/ADEChallengeData2016/tfrecord"

class dataset:
    name = "pascal_voc_seg"
    train = 1464       #number of training images
    train_aug = 10582
    trainval = 2913
    val = 1449
    classes = 21           #classes including ignore_label
    ignore_label = 255        #label that does not participate training and inference
    train_steps = 30000
    data_dir = "./data/tfrecord"#"../../dataSet/pascal_voc_seg/tfrecord"

def quantize_grads(grads_and_vars,model_class,lrate):
    grads = []
    for grad_and_var in grads_and_vars:
        grads.append([QG(grad_and_var[0],lrate),grad_and_var[1]])
    return grads

def train(loss_val,model_class,lrate):
    optimizer = tf.train.GradientDescentOptimizer(1)
    grads = optimizer.compute_gradients(loss_val)
    grads = quantize_grads(grads,model_class,lrate)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_op += model_class.W_clip_op
    with tf.control_dependencies(update_op):
        train_op = optimizer.apply_gradients(grads)

    return train_op

tf.logging.set_verbosity(tf.logging.INFO)

print("Setting up dataset reader")
with tf.device('/cpu:0'):
    data_train = segmentation_dataset.get_dataset(
            dataset.name,"train_aug",dataset.data_dir)#training for ade, train_aug for pascal
    data_val = segmentation_dataset.get_dataset(
            dataset.name,"val",dataset.data_dir)#validation for ade, val for pascal
    batchTrain = input_generator.get(
        data_train,
        args.crop_size,
        args.batchSize,#is_training=True,
        dataset_split="training")
    batchTest = input_generator.get(
        data_val,
        args.crop_size,
        args.testBatchSize,#is_training=False,
        dataset_split="validation")
    train_images = batchTrain['image']
    print(train_images)
    train_annotations = batchTrain['label']
    print(train_annotations)
    valid_images = batchTest['image']
    valid_annotations = batchTest['label']
    valid_names = batchTest['image_name']
    valid_height = batchTest['height']
    valid_width = batchTest['width']

is_training = tf.Variable(initial_value=args.is_training, trainable=False, name='train_stat', dtype=tf.bool)

#setting up the network
model = Deeplab_v3(dataset.classes,batch_norm_decay=args.batch_norm_decay,is_training=is_training)
logits = model.forward_pass(train_images)
predicts = tf.argmax(logits, axis=-1, name='predicts')

variables_to_restore = tf.trainable_variables(scope='resnet_v2_50')

# finetune resnet_v2_50的参数(block1到block4)
restorer = tf.train.Saver(variables_to_restore)

cross_entropy = loss(logits,train_annotations,dataset.classes,ignore_label=dataset.ignore_label)
# l2_norm l2正则化
l2_loss = args.weight_decay * tf.add_n(
     [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
loss = cross_entropy + l2_loss
tf.summary.scalar("loss",loss)

lr = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
train_op =train(loss,model,lr)

#train benchmark
out = tf.reshape(tf.argmax(logits,axis=3),shape=[-1])#[B,H,W]
labels = tf.reshape(tf.cast(tf.squeeze(train_annotations,squeeze_dims=[3]),dtype=tf.int64), shape=[-1])

indices = tf.squeeze(tf.where(tf.logical_not(tf.equal(
        labels, dataset.ignore_label))), 1)
labels = tf.cast(tf.gather(labels, indices), tf.int64)
out = tf.gather(out, indices)
accuracy = tf.reduce_mean(tf.cast(tf.logical_not(tf.equal(out, labels)), tf.float32))
tf.summary.scalar("accuracy",accuracy)

#test benchmark
tf.get_variable_scope().reuse_variables()
valid_logits = model.forward_pass(valid_images)
# valid_logits = test_aug(valid_images,model)
valid_output = tf.argmax(valid_logits, axis=-1, name='predicts')# used to visualize the prediciton
valid_predicts = tf.reshape(tf.argmax(valid_logits, axis=-1, name='predicts'), shape=[-1])
valid_labels = tf.reshape(tf.cast(tf.squeeze(valid_annotations,squeeze_dims=[3]),dtype=tf.int64), shape=[-1])

valid_indices = tf.squeeze(tf.where(tf.logical_not(tf.equal(
        valid_labels, dataset.ignore_label))), 1)
valid_labels = tf.cast(tf.gather(valid_labels, valid_indices), tf.int64)
valid_predicts = tf.gather(valid_predicts, valid_indices)

valid_accuracy = tf.reduce_mean(tf.cast(tf.logical_not(tf.equal(valid_predicts, valid_labels)), tf.float32))

mean_iou_val, conf_mat_val = tf.metrics.mean_iou(valid_predicts,valid_labels,dataset.classes,name="miou")

# 我们要保存所有的参数
saver = tf.train.Saver(tf.all_variables())

running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="miou")
running_vars_initializer = tf.variables_initializer(var_list=running_vars)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)#start the queue runner
    if "train" in mode:
        # finetune resnet_v2_50参数
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Last check point restored...")
        else:
            restorer.restore(sess, '../ckpts/resnet_v2_50.ckpt')
            sess.run([model.W_q_op])
            print("Model restored, weights quantized.")

    elif mode=="val" or mode=="vis":
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Last check point restored...")
        else:
            print("Model not fond!")
    else:
        print("This mode is illeagal, please check!")

    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    if not os.path.exists(sums_dir): os.makedirs(sums_dir)
    if not os.path.exists(loss_dir): os.makedirs(loss_dir)
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)
    #-------------------------------TensorBoard--------------------------------------
    #setting up tensorboard summary
    train_writer = tf.summary.FileWriter(sums_dir, sess.graph)
    merge = tf.summary.merge_all()


    if mode == "train":
        lastAccu = 1.
        thisAccu = 1.
        with open(loss_dir + 'loss_curve.csv', 'w') as f:
            writer = csv.writer(f)
            for step in range(1, dataset.train_steps):
                # update learning rate
                if len(args.lr_schedule) / 2:
                  if step == args.lr_schedule[0]:
                    args.lr_schedule.pop(0)
                    lr_new = args.lr_schedule.pop(0)
                    lr_old = sess.run(lr)
                    sess.run(lr.assign(lr_new))
                    tf.logging.info('lr: %f -> %f' % (lr_old, lr_new))
                #train and evaluate
                loss_tr, l2_loss_tr, predicts_tr, accu, _ = sess.run(
                    fetches=[cross_entropy, l2_loss, predicts, accuracy, train_op])
                #display training loss and accuracy
                if step % args.display == 0:
                    tf.logging.info('Step:%s , loss:%.4f, accuracy:%.4f' % (step,loss_tr, accu))
                #-------------------------------TensorBoard--------------------------------------
                #write summary
                if (step+1) % args.write_summary == 0:
                    summary = sess.run(merge)
                    train_writer.add_summary(summary, step)

                if (step+1) % args.try_save == 0:
                    batchNumTest = dataset.val // args.testBatchSize
                    sess.run(is_training.assign(False))
                    sess.run(running_vars_initializer)
                    for val_step in range(dataset.val // args.testBatchSize):
                        sess.run(conf_mat_val)
                        score = sess.run(mean_iou_val)
                        # mean_score = np.sum(score)/dataset.classes
                    #tf.logging.info('mIoU on valid set:%.4f' % (mean_score))
                    tf.logging.info('mIoU on valid set:%.4f' % (score))
                    saver.save(sess, logs_dir + "model.ckpt", step)
                    sess.run(is_training.assign(True))
                writer.writerow([step,loss_tr,accu])

    #code to evaluate the model
    elif mode == "val":
        total_accu = 0.
        for step in range(dataset.val // args.testBatchSize):
            sess.run(conf_mat_val)
            mean_score,accu = sess.run([mean_iou_val,valid_accuracy])
            # mean_score = np.sum(score)/dataset.classes
            tf.logging.info('Error on step %s is:%.4f, mIoU is: %.4f' % (step,accu,mean_score))
            total_accu += accu
        total_accu /= dataset.val // args.testBatchSize
        tf.logging.info('mIoU on valid set:%.4f ,error:%.4f' % (mean_score,total_accu))

    #code to visualize the results
    elif mode == "vis":
        if not os.path.exists(vis_dir + "picture"): os.makedirs(vis_dir + "picture")
        if not os.path.exists(vis_dir + "ground"): os.makedirs(vis_dir + "ground")
        if not os.path.exists(vis_dir + "predict"): os.makedirs(vis_dir + "predict")
        for step in range(dataset.val // args.testBatchSize):
            image, anno, pred, name, height, width= sess.run([valid_images,
                    valid_annotations,valid_output, valid_names,
                     valid_height, valid_width])
            anno = np.squeeze(anno, axis=3)
            for itr in range(args.testBatchSize):
                image_save = image[itr, :height[itr], :width[itr],0:3]
                anno_save = anno[itr, :height[itr], :width[itr]]
                pred_save = pred[itr, :height[itr], :width[itr]]
                # print(image_save.shape)
                misc.imsave(os.path.join(vis_dir + "ground", name[itr] + ".png"), anno_save.astype(np.uint8))
                misc.imsave(os.path.join(vis_dir + "picture", name[itr] + ".png"), image_save.astype(np.uint8))
                misc.imsave(os.path.join(vis_dir + "predict", name[itr] + ".png"), pred_save.astype(np.uint8))
                print("Saved image: %s" % str(step*args.testBatchSize+itr))
