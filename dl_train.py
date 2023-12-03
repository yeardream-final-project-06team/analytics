from time import time
from scipy.sparse import csc_matrix
import tensorflow as tf
import numpy as np
import h5py
import pandas as pd
from collections import defaultdict




###########################  Hyper Parameter  ##################################

# Common hyperparameter settings
n_hid = 500
n_dim = 5
n_layers = 2
gk_size = 3

# Different hyperparameter settings for each dataset
lambda_2 = 20.  # l2 regularisation
lambda_s = 0.006
iter_p = 5  # optimisation
iter_f = 5
epoch_p = 30  # training epoch
epoch_f = 60
dot_scale = 1  # scaled dot product




###########################  Load Data  ##################################

def load_train_data(data:pd.DataFrame):
    id_map = defaultdict(lambda:len(id_map))
    persona_map = defaultdict(lambda:len(persona_map))

    for vid, persona in data[['id', 'persona']].itertuples(index=False):
        id_map[vid]
        persona_map[persona]

    data['vid'] = data['id'].replace(id_map)
    data['pid'] = data['persona'].replace(persona_map)

    total = data[['vid', 'pid', 'title']]
    total = total.groupby(['vid','pid']).count()

    train = total.sample(frac=0.7, random_state=777)
    test = total.loc[[True if i not in train.index else False for i in total.index ], :]


    train = train.reset_index().to_numpy().astype('int32')
    test = test.reset_index().to_numpy().astype('int32')
    total = total.reset_index().to_numpy().astype('int32')

    n_u = np.unique(total[:,0]).size  # num of users
    n_m = np.unique(total[:,1]).size  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_r[train[i,1]-1, train[i,0]-1] = train[i,2]

    for i in range(n_test):
        test_r[test[i,1]-1, test[i,0]-1] = test[i,2]

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m


###########################  Network Function  ##################################
def local_kernel(u, v):

    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist**2)

    return hat


def kernel_layer(x, n_hid=n_hid, n_dim=n_dim, activation=tf.nn.sigmoid, lambda_s=lambda_s, lambda_2=lambda_2, name=''):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-3))
        b = tf.get_variable('b', [n_hid])

    w_hat = local_kernel(u, v)
    
    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])
    
    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    W_eff = W * w_hat  # Local kernelised weight matrix
    y = tf.matmul(x, W_eff) + b
    y = activation(y)

    return y, sparse_reg_term + l2_reg_term


def global_kernel(input, gk_size, dot_scale):

    avg_pooling = tf.reduce_mean(input, axis=1)  # Item (axis=1) based average pooling
    avg_pooling = tf.reshape(avg_pooling, [1, -1])
    n_kernel = avg_pooling.shape[1].value

    conv_kernel = tf.get_variable('conv_kernel', initializer=tf.random.truncated_normal([n_kernel, gk_size**2], stddev=0.1))
    gk = tf.matmul(avg_pooling, conv_kernel) * dot_scale  # Scaled dot product
    gk = tf.reshape(gk, [gk_size, gk_size, 1, 1])

    return gk


def global_conv(input, W):

    input = tf.reshape(input, [1, input.shape[0], input.shape[1], 1])
    conv2d = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME'))

    return tf.reshape(conv2d, [conv2d.shape[1], conv2d.shape[2]])



###########################  Network Instantiation  ##################################

def pretraining(R, train_m, train_r, n_u):
    y = R
    reg_losses = None

    for i in range(n_layers):
        y, reg_loss = kernel_layer(y, name=str(i))
        reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

    pred_p, reg_loss = kernel_layer(y, n_u, activation=tf.identity, name='out')
    reg_losses = reg_losses + reg_loss

    # L2 loss
    diff = train_m * (train_r - pred_p)
    sqE = tf.nn.l2_loss(diff)
    loss_p = sqE + reg_losses

    optimizer_p = tf.contrib.opt.ScipyOptimizerInterface(loss_p, options={'disp': True, 'maxiter': iter_p, 'maxcor': 10}, method='L-BFGS-B')

    return pred_p, optimizer_p


def fine_tuning(R, train_m, train_r, n_u):
    y = R
    reg_losses = None

    for i in range(n_layers):
        y, _ = kernel_layer(y, name=str(i))

    y_dash, _ = kernel_layer(y, n_u, activation=tf.identity, name='out')

    gk = global_kernel(y_dash, gk_size, dot_scale)  # Global kernel
    y_hat = global_conv(train_r, gk)  # Global kernel-based rating matrix

    for i in range(n_layers):
        y_hat, reg_loss = kernel_layer(y_hat, name=str(i))
        reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

    pred_f, reg_loss = kernel_layer(y_hat, n_u, activation=tf.identity, name='out')
    reg_losses = reg_losses + reg_loss

    # L2 loss
    diff = train_m * (train_r - pred_f)
    sqE = tf.nn.l2_loss(diff)
    loss_f = sqE + reg_losses

    optimizer_f = tf.contrib.opt.ScipyOptimizerInterface(loss_f, options={'disp': True, 'maxiter': iter_f, 'maxcor': 10}, method='L-BFGS-B')

    return pred_f, optimizer_f



###########################  Evaluation  ##################################

def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg


def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num



###########################  Traiing and Test Loop  ##################################

def train(R, train_m, train_r, test_m, test_r, optimizer_p, optimizer_f, pred_p, pred_f):
    best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
    best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0

    time_cumulative = 0
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch_p):
            tic = time()
            optimizer_p.minimize(sess, feed_dict={R: train_r})
            pre = sess.run(pred_p, feed_dict={R: train_r})

            t = time() - tic
            time_cumulative += t
            
            error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
            test_rmse = np.sqrt(error)

            error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
            train_rmse = np.sqrt(error_train)

            print('.-^-._' * 12)
            print('PRE-TRAINING')
            print('Epoch:', i+1, 'test rmse:', test_rmse, 'train rmse:', train_rmse)
            print('Time:', t, 'seconds')
            print('Time cumulative:', time_cumulative, 'seconds')
            print('.-^-._' * 12)

        for i in range(epoch_f):
            tic = time()
            optimizer_f.minimize(sess, feed_dict={R: train_r})
            pre = sess.run(pred_f, feed_dict={R: train_r})

            t = time() - tic
            time_cumulative += t
            
            error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
            test_rmse = np.sqrt(error)

            error_train = (train_m * (np.clip(pre, 1., 5.) - train_r) ** 2).sum() / train_m.sum()  # train error
            train_rmse = np.sqrt(error_train)

            test_mae = (test_m * np.abs(np.clip(pre, 1., 5.) - test_r)).sum() / test_m.sum()
            train_mae = (train_m * np.abs(np.clip(pre, 1., 5.) - train_r)).sum() / train_m.sum()

            test_ndcg = call_ndcg(np.clip(pre, 1., 5.), test_r)
            train_ndcg = call_ndcg(np.clip(pre, 1., 5.), train_r)

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_rmse_ep = i+1

            if test_mae < best_mae:
                best_mae = test_mae
                best_mae_ep = i+1

            if best_ndcg < test_ndcg:
                best_ndcg = test_ndcg
                best_ndcg_ep = i+1

            print('.-^-._' * 12)
            print('FINE-TUNING')
            print('Epoch:', i+1, 'test rmse:', test_rmse, 'test mae:', test_mae, 'test ndcg:', test_ndcg)
            print('Epoch:', i+1, 'train rmse:', train_rmse, 'train mae:', train_mae, 'train ndcg:', train_ndcg)
            print('Time:', t, 'seconds')
            print('Time cumulative:', time_cumulative, 'seconds')
            print('.-^-._' * 12)

        saver = tf.train.Saver()
        saver.save(sess, 'dl_model')


    # Final result
    print('Epoch:', best_rmse_ep, ' best rmse:', best_rmse)
    print('Epoch:', best_mae_ep, ' best mae:', best_mae)
    print('Epoch:', best_ndcg_ep, ' best ndcg:', best_ndcg)


if __name__=='__main__':
    data = pd.read_csv('video_detail.csv')
    n_m, n_u, train_r, train_m, test_r, test_m = load_train_data(data)

    R = tf.placeholder("float", [n_m, n_u])

    pred_p, optimizer_p = pretraining(R, train_m, train_r, n_u)
    pred_f, optimizer_f = fine_tuning(R, train_m, train_r, n_u)

    train(R, train_m, train_r, test_m, test_r, optimizer_p, optimizer_f, pred_p, pred_f)