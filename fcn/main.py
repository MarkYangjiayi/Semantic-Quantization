#coding=utf-8
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import scipy.misc as misc
import os
import sys
import fcn8_vgg
from loss import loss
sys.path.append('../data_utils')
import segmentation_dataset
import input_generator
from quant_test import QG
import csv
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "4"#default should be 1

note = "9-06-int8"#8888_f_new
mode = "train"#train, val or vis
logs_dir="../save/" + note + "/ckpt/"#change this when you train a new model
sums_dir="../save/" + note + "/summary/"
loss_dir="../save/" + note + "/loss/"
vis_dir="../save/" + note + "/vis/"

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

class args:
    vgg16_path = "../vgg16.npy"
    write_summary = 1000
    try_save = 1000
    train_steps = 30000
    batchSize = 4
    testBatchSize = 1
    crop_size = [512,512]   #size to crop, automatically padded
    #lr_schedule = [1,0.001,10000,0.0005,20000,0.00025]#qg3.1
    #lr_schedule = [1,0.001,10000,0.0005,20000,0.00025]#qg3
    #lr_schedule = [1,0.001,2000,0.0005,5000,0.00025,10000,0.00001]#qg2
    #lr_schedule = [1, 0.0625, 10000, 0.03125, 24000, 0.015625]#qg1.2 for U18
    #lr_schedule = [1, 0.0625, 10000, 0.03125]#qg1.1
    lr_schedule = [1, 0.0625, 10000, 0.03125, 20000, 0.015625]#qg1 this is the official learning rate
    #lr_schedule = [1, 0.0625, 50000, 0.03125, 100000, 0.015625, 150000, 0.0078125]#lr2 off learning rate for ADE20K
    #lr_schedule = [1, 0.0625, 50000, 0.03125, 100000, 0.015625, 150000, 0.0078125]#lr1
    #lr_schedule = [1, 8, 10000, 4, 20000, 2]#quantized training
    # lr_schedule = [1, 2e-5, 4000, 1e-5, 8000, 2e-6, 10000, 1e-6]#float training
    # lr_schedule = [1, 2e-5, 4000, 1e-5, 15000, 2e-6]#float training
    # lr_schedule = [1, 0.001]
    #lr_schedule = [1, 0.001, 50000, 0.0005, 100000, 0.0001, 120000, 0.00005,150000,0.00001]#learning_rate 2
    # lr_schedule = [1, 0.001, 5000, 0.0005, 10000, 0.00025,15000, 0.0001,25000, 0.00005]#learning rate for pascal
    is_training = True

# class dataset:
#     name = "ade20k"
#     train = 20210       #number of training images
#     trainaug = 0
#     trainval = 0
#     val = 2000
#     classes = 151           #classes including ignore_label
#     ignore_label = 0        #label that does not participate training and inference
#     data_dir = "../../dataSet/ADEChallengeData2016/tfrecord"

class dataset:
    name = "pascal_voc_seg"
    train = 1464       #number of training images
    train_aug = 10582
    trainval = 2913
    val = 1449
    classes = 21           #classes including ignore_label
    ignore_label = 255        #label that does not participate training and inference
    data_dir = "../../dataSet/pascal_voc_seg/tfrecord"

# img1 = scp.misc.imread("./test_data/tabby_cat.png")
def quantize_grads(grads_and_vars,vgg_fcn,lr):
    grads = []
    for grad_and_var in grads_and_vars:
        print(grad_and_var[0].op.name)
        # tf.summary.histogram(grad_and_var[0].op.name,grad_and_var[0])
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_op += vgg_fcn.W_clip_op
        with tf.control_dependencies(update_op):
            if grad_and_var[1].name.find('conv1_1')>-1 or grad_and_var[1].name.find('upscore32')>-1:
                grads.append([QG(grad_and_var[0],lr),grad_and_var[1]])
                #grads.append([grad_and_var[0]*lr,grad_and_var[1]])
            else:
                grads.append([QG(grad_and_var[0],lr),grad_and_var[1]])
    return grads

def train(loss_val,vgg_fcn):
    # optimizer = tf.train.AdamOptimizer(lr)
    optimizer = tf.train.GradientDescentOptimizer(1)
    # optimizer = tf.train.MomentumOptimizer(lr,0.75)
    grads = optimizer.compute_gradients(loss_val)
    grads = quantize_grads(grads,vgg_fcn,lr)
    return optimizer.apply_gradients(grads)

#load data
with tf.device('/cpu:0'):
        print("Setting up dataset reader")
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
        train_annotations = batchTrain['label']#[N,H,W,1]
        valid_images = batchTest['image']
        valid_annotations = batchTest['label']
        valid_names = batchTest['image_name']
        valid_height = batchTest['height']
        valid_width = batchTest['width']

is_training = tf.Variable(initial_value=args.is_training, trainable=False, name='train_stat', dtype=tf.bool)

#Setting up network
vgg_fcn = fcn8_vgg.FCN8VGG(args.vgg16_path)
logits= vgg_fcn.build(train_images,num_classes=dataset.classes,train=is_training, debug=False)#upscore32 = [N,H,W,151]

loss_func = loss(logits,train_annotations,dataset.classes,ignore_label=dataset.ignore_label)
tf.summary.scalar("loss",loss_func)

lr = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
train_op =train(loss_func,vgg_fcn)

#train benchmark
out = tf.reshape(tf.argmax(logits,axis=3), shape=[-1])
labels = tf.reshape(tf.cast(tf.squeeze(train_annotations,squeeze_dims=[3]),dtype=tf.int64), shape=[-1])

indices = tf.squeeze(tf.where(tf.logical_not(tf.equal(
        labels, dataset.ignore_label))), 1)
labels = tf.cast(tf.gather(labels, indices), tf.int64)
out = tf.gather(out, indices)
accuracy = tf.reduce_mean(tf.cast(tf.logical_not(tf.equal(out, labels)), tf.float32))
tf.summary.scalar("accuracy",accuracy)

#test benchmark
tf.get_variable_scope().reuse_variables()
valid_logits= vgg_fcn.build(valid_images,num_classes=dataset.classes,train=is_training, debug=False)#upscore32 = [N,H,W,151]
valid_output = tf.argmax(valid_logits, axis=-1, name='predicts')# used to visualize the prediciton
valid_predicts = tf.reshape(tf.argmax(valid_logits, axis=-1, name='predicts'), shape=[-1])
valid_labels = tf.reshape(tf.cast(tf.squeeze(valid_annotations,squeeze_dims=[3]),dtype=tf.int64), shape=[-1])

valid_indices = tf.squeeze(tf.where(tf.logical_not(tf.equal(
        valid_labels, dataset.ignore_label))), 1)
valid_labels = tf.cast(tf.gather(valid_labels, valid_indices), tf.int64)
valid_predicts = tf.gather(valid_predicts, valid_indices)

valid_accuracy = tf.reduce_mean(tf.cast(tf.logical_not(tf.equal(valid_predicts, valid_labels)), tf.float32))

mean_iou_val, conf_mat_val = tf.metrics.mean_iou(valid_predicts,valid_labels,dataset.classes,name="miou")

saver = tf.train.Saver()

running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="miou")
running_vars_initializer = tf.variables_initializer(var_list=running_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())# have to exist in order to compute mean_iou
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)#start the queue runner

    if "train" in mode:
        #Restore checkpoint if exist
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            result = sess.run(vgg_fcn.W_q_op)
            #print(result[0])
            print("Weights quantized...")

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
        with open(loss_dir + 'loss_curve.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Step","Loss"])
            for step in range(1, args.train_steps):
                #update learning rate
                if len(args.lr_schedule) / 2:
                  if step == args.lr_schedule[0]:
                    args.lr_schedule.pop(0)
                    lr_new = args.lr_schedule.pop(0)
                    lr_old = sess.run(lr)
                    sess.run(lr.assign(lr_new))
                    print ('lr: %f -> %f' % (lr_old, lr_new))
                _, train_loss, accu = sess.run([train_op,loss_func,accuracy])
                logging.info('Step: %03d Train: %.4f Accuracy:%.4f' % (step, train_loss, accu))
                if train_loss is not None:
                    writer.writerow([step,train_loss,accu])

                # if (step+1) % args.write_summary == 0 or (step+1) == 10:
                #     # summary = sess.run(merge)
                #     # train_writer.add_summary(summary, step)
                #
                #     # target_list = []
                #     # for targets in tf.get_collection("quantized_weights"):
                #     #     target_list.append(targets.name)
                #     #     print(targets.name)
                #     # print("Finished")
                #     # print(target_list)
                #
                #
                #     extract_distri(sess,step)


                if (step+1) % args.try_save == 0:
                    # saver.save(sess, logs_dir + "model.ckpt", step)
                    batchNumTest = dataset.val // args.testBatchSize
                    sess.run(is_training.assign(False))
                    sess.run(running_vars_initializer)
                    for step in range(dataset.val // args.testBatchSize):
                        sess.run(conf_mat_val)
                        score = sess.run([mean_iou_val])
                        mean_score = np.sum(score)/dataset.classes
                    tf.logging.info('mIoU on valid set:%.4f' % (mean_score))
                    saver.save(sess, logs_dir + "model.ckpt", step)
                    sess.run(is_training.assign(True))
            logging.info("Finished training...")

    #code to evaluate the model
    elif mode == "val":
        total_accu = 0.
        for step in range(dataset.val // args.testBatchSize):
            sess.run(conf_mat_val)
            score,accu = sess.run([mean_iou_val,valid_accuracy])
            mean_score = np.sum(score)/dataset.classes
            logging.info('Error on step %s is:%.4f, mIoU is: %.4f' % (step,accu,mean_score))
            total_accu += accu
        total_accu /= dataset.val // args.testBatchSize
        logging.info('mIoU on valid set:%.4f ,error:%.4f' % (mean_score,total_accu))
        logging.info(score)

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
