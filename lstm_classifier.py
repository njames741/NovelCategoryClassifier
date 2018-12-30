# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import json
import numpy as np
from gensim.models import word2vec
from gensim import models
import logging
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#參數
lr = 0.0005 # learning rate
batch_size = 8
n_inputs = 400 # 28 cl
n_steps = None # 28 rows -> time stamps
n_hidden_unins = 70 # hidden units
n_classes = 7
novel_take_num = 500
repeat_time = 100
logs_path = "TensorBoard/"
model_path = "./tmp/model.ckpt"
train_path = "segResult/train"
test_path = "segResult/test"

def gen_train():
    model = models.Word2Vec.load('../w2vModel/med250.model.bin')
    class_num = 0
    for subdir in os.listdir(train_path):
        if subdir == "其它小說" or subdir == "遊戲競技" or subdir == "歷史軍事" or subdir == "科幻靈異":
            continue
        dirName = os.path.join(train_path, subdir)
        novel_num = 0
        for file in os.listdir(os.path.join(train_path, subdir)):
            fileName = os.path.join(dirName, file)
            with open(fileName,'r') as load_f:
                load_dict = json.load(load_f)
                novel_vector = []
                for i in load_dict[u'內容'][6].split(' '):
                    try:
                        novel_vector.append(model[i])
                    except Exception as e:
                        pass

                yield (load_dict[u'類別'], novel_vector)

            if novel_num == novel_take_num:
                break
            novel_num += 1


def gen_test():
    model = models.Word2Vec.load('../w2vModel/med250.model.bin')
    class_num = 0
    for subdir in os.listdir(test_path):
        if subdir == "其它小說" or subdir == "遊戲競技" or subdir == "歷史軍事" or subdir == "科幻靈異":
            continue
        dirName = os.path.join(test_path, subdir)
        novel_num = 0
        for file in os.listdir(os.path.join(test_path, subdir)):
            fileName = os.path.join(dirName, file)
            with open(fileName,'r') as load_f:
                load_dict = json.load(load_f)
                novel_vector = []
                for i in load_dict[u'內容'][7].split(' '):
                    try:
                        novel_vector.append(model[i])
                    except Exception as e:
                        pass
                yield (load_dict[u'類別'], novel_vector)
            
            if novel_num == novel_take_num:
                break
            novel_num += 1


#製作training dataset
dataset = tf.data.Dataset.from_generator(gen_train,(tf.float64, tf.float64), (tf.TensorShape([n_classes]),tf.TensorShape(None)) )
dataset = dataset.shuffle(buffer_size=3800).padded_batch(batch_size, padded_shapes=([None],[None,400]))
dataset = dataset.repeat(repeat_time)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

#製作testing dataset
dataset_test = tf.data.Dataset.from_generator(gen_test,(tf.float64, tf.float64), (tf.TensorShape([n_classes]),tf.TensorShape(None)) )
dataset_test = dataset_test.shuffle(buffer_size=300).padded_batch(300, padded_shapes=([None],[None,400]))
iterator_test = dataset_test.make_one_shot_iterator()
one_element_test = iterator_test.get_next()

#建構網路
xs =tf.placeholder(tf.float32, [None, None, n_inputs], name="inputs")
ys =tf.placeholder(tf.float32, [None, n_classes], name="outputs")
weights = {
    'in': tf.Variable(tf.random_uniform([n_inputs, n_hidden_unins], -1.0, 1.0), name="in_w"),
    'out': tf.Variable(tf.random_uniform([n_hidden_unins, n_classes], -1.0, 1.0), name="out_w"),
}
b = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unins]), name="in_bias"),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name="out_bias"),
}

def RNN(X, weights, bias):
    def LstmCell():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unins)
        return lstm_cell
    def LstmCell1():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(300)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.6)
        return lstm_cell
    def LstmCell2():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(150)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.6)
        return lstm_cell
    def LstmCell3():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(70)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.6)
        return lstm_cell


    with tf.name_scope("RNN_CELL"):
        lstm_multi = tf.nn.rnn_cell.MultiRNNCell([LstmCell1(),LstmCell2(),LstmCell3()],  state_is_tuple=True) 
        #此時，得到的outputs就是time_steps步裏所有的輸出。它的形狀爲(batch_size, time_steps, cell.output_size)。
        #state是最後一步的隱狀態，tuple格式(c,h)，它的形狀爲(batch_size, cell.state_size)。states.shape = [layer_num, 2, batch_size, hidden_size]
        #output[:, -1, :] = states[-1].h       states[-1]代表最後一層LSTM的states
        outputs, states = tf.nn.dynamic_rnn(lstm_multi, X, dtype=tf.float32)

    with tf.name_scope('outlayer'):
        results = tf.matmul(states[-1].h, weights['out']) + b['out']
        ###GRUcell###
        # outputs = tf.transpose(outputs, [1, 0, 2])
        # last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
        # results = tf.matmul(outputs[-1], weights['out']) + b['out']
        ###GRUcell###
    return results

pred = RNN(xs, weights, b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=ys))
tf.summary.scalar("Loss", cost)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()


if __name__ == '__main__':
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allocator_type = 'BFC'
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config = config) as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

        start_time = time.time()
        try:
            round = 0
            while True:
                batch_y,batch_x = sess.run(one_element)
                print(batch_x.shape)
                sess.run(train_op, feed_dict={xs: batch_x, ys: batch_y})
                acc, loss = sess.run([accuracy, cost], feed_dict={xs: batch_x, ys: batch_y})
                print("Step " + str(round) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                summary = sess.run(merged, feed_dict = {xs: batch_x, ys: batch_y})
                writer.add_summary(summary, round)

                print("*********",round,"**********")
                round = round + 1
        except tf.errors.OutOfRangeError:
            print("end!")
            end_time = time.time()
            saver.save(sess, model_path)


        test_y, test_x = sess.run(one_element_test)
        acc = sess.run(accuracy, feed_dict = {xs: test_x, ys: test_y})

        print("model save in" + model_path)
        print("time: ",end_time-start_time)
        print("Testing Accuracy = " + "{:.5f}".format(acc))
        print("novel_Num : ", novel_take_num)
        print("Batch size : ", batch_size)
        print("Learning rate : ", lr)
        print("model_path : ", model_path)
        print("Repeat time : ", repeat_time)