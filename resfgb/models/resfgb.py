# coding : utf-8

"""
ResFGB for multiclass classificcation problems.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
from logging import getLogger, ERROR
import time
from tqdm import tqdm
import sys
import numpy as np
import theano
from resfgb.models import LogReg, SVM, ResGrad

logger = getLogger(__name__)


class ResFGB(object):
    def __init__(self, model_type=u'logreg', model_hparams={}, resblock_hparams={},
                 fg_eta=None, max_iters=10, seed=99, proc_batch_size=10000):

        self.show_param(model_type,
                        model_hparams['tune_eta'],
                        model_hparams['max_epoch'],
                        model_hparams['early_stop'],
                        max_iters)

        self.__tune_eta__ = model_hparams['tune_eta']
        self.__max_epoch__ = model_hparams['max_epoch']
        self.__early_stop__ = model_hparams['early_stop']
        del model_hparams['tune_eta']
        del model_hparams['max_epoch']
        del model_hparams['early_stop']

        if model_type == u'logistic':
            self.__model__ = LogReg(seed=seed, **model_hparams)
        elif model_type == u'smooth_hinge':
            self.__model__ = SVM(seed=seed, **model_hparams)
        else:
            logger.log(ERROR, 'invalid model_type: {0}'.format(model_type))
            sys.exit(-1)

        self.__max_iters__ = max_iters
        self.__fg__ = ResGrad(self.__model__, eta=fg_eta,
                              resblock_hparams=resblock_hparams,
                              seed=seed, proc_batch_size=proc_batch_size)

    def show_param(self, model_type, tune_eta, max_epoch, early_stop, max_iters):
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'ResFGB setting', '-' * 5))
        logger.info('{0:<15}{1:>21}'.format('model_type', model_type))
        logger.info('{0:<15}{1:>21}'.format('tune_eta', tune_eta))
        logger.info('{0:<15}{1:>21}'.format('max_epoch', max_epoch))
        logger.info('{0:<15}{1:>21}'.format('early_stop', early_stop))
        logger.info('{0:<15}{1:>21}'.format('max_iters', max_iters))

    def evaluate(self, X, Y, sample_f=True, eval_metric=None):
        if sample_f:
            Z = self.__fg__.predict(X)
            loss, acc = self.__model__.evaluate(Z, Y, eval_metric=eval_metric)
        else:
            loss, acc = self.__model__.evaluate(X, Y, eval_metric=eval_metric)

        return loss, acc

    def predict(self, X, sample_f=True):
        if sample_f:
            Z = self.__fg__.predict(X)
        else:
            Z = X
        pred = self.__model__.predict(Z)
        return pred

    def predict_proba(self, X, sample_f=True):
        if sample_f:
            Z = self.__fg__.predict(X)
        else:
            Z = X
        pred = self.__model__.predict_proba(Z)
        return pred

    def fit(self, X, Y, Xv=None, Yv=None, use_best_iter=False, eval_metric=None):

        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'Training ResFGB', '-' * 5))

        best_val_acc = None
        best_val_loss = 1e+10
        best_param = None
        best_n_layers = None
        total_time = 0.

        Z = np.array(X)

        if Xv is not None:
            monitor = True
            Zv = np.array(Xv)
        else:
            monitor = False
            Zv = None

        for n_iter in range(self.__max_iters__):
            logger.info('resfgb epoch: %s / %s' % (n_iter, self.__max_iters__))
            # ----- apply functional gradient -----
            stime = time.time()
            if n_iter >= 1:
                Z = self.__fg__.apply(Z, lfrom=n_iter - 1)
                if monitor:
                    Zv = self.__fg__.apply(Zv, lfrom=n_iter - 1)

            # ----- fit and evaluate -----
            self.__model__.optimizer.reset_func()
            if self.__tune_eta__ and (n_iter == 0):
                self.__model__.determine_eta(Z, Y)

            self.__model__.fit(Z, Y, self.__max_epoch__, early_stop=self.__early_stop__)

            etime = time.time()
            total_time += etime - stime

            train_loss, train_acc = self.evaluate(Z, Y, sample_f=False, eval_metric=eval_metric)
            logger.info('layer: {0:4}, time:{1:>14.1f} sec'
                        .format(n_iter, total_time))
            logger.info('train_loss: {0:5.4f}, train_acc: {1:4.3f}'
                        .format(train_loss, train_acc))

            if monitor:
                val_loss, val_acc = self.evaluate(Zv, Yv, sample_f=False, eval_metric=eval_metric)
                logger.info('val_loss: {0:8.4f}, val_acc: {1:7.3f}'
                            .format(val_loss, val_acc))

                if best_val_acc is None or val_acc > best_val_acc:
                    best_n_layers = n_iter
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_param = self.__model__.get_params(real_f=True)

            # ----- compute weight matrix -----
            stime = time.time()
            self.__fg__.compute_weight(Z, Y)
            etime = time.time()
            total_time += etime - stime

        # ----- apply functional gradient -----
        stime = time.time()
        if self.__max_iters__ >= 1:
            Z = self.__fg__.apply(Z, lfrom=self.__max_iters__ - 1)
            if monitor:
                Zv = self.__fg__.apply(Zv, lfrom=self.__max_iters__ - 1)

        # ----- fit and evaluate -----
        self.__model__.optimizer.reset_func()
        self.__model__.fit(Z, Y, self.__max_epoch__, early_stop=self.__early_stop__)

        etime = time.time()
        total_time += etime - stime

        train_loss, train_acc = self.evaluate(Z, Y, sample_f=False, eval_metric=eval_metric)
        logger.info('layer: {0:4}, time:{1:>14.1f} sec'
                    .format(self.__max_iters__, total_time))
        logger.info('train_loss: {0:5.4f}, train_acc: {1:4.3f}'
                    .format(train_loss, train_acc))

        if monitor:
            val_loss, val_acc = self.evaluate(Zv, Yv, sample_f=False, eval_metric=eval_metric)
            logger.info('val_loss: {0:8.4f}, val_acc: {1:7.3f}'
                        .format(val_loss, val_acc))

            if val_acc > best_val_acc:
                best_n_layers = self.__max_iters__
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_param = self.__model__.get_params(real_f=True)

        # ----- finalize -----
        if monitor and use_best_iter is True:
            if best_n_layers < self.__max_iters__:
                del self.__fg__.params[best_n_layers:]
                self.__model__.set_params(best_param)

        if monitor:
            if use_best_iter is True:
                return (best_n_layers, best_val_loss, best_val_acc)
            else:
                return (self.__max_iters__, val_loss, val_acc)
        else:
            return (None, None, None)
