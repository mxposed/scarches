import tensorflow as tf
from keras import backend as K

from ._utils import compute_mmd, _nelem, _nan2zero, _nan2inf, _reduce_mean


def kl_recon(mu, log_var, alpha=0.1, eta=1.0):
    def kl_recon_loss(y_true, y_pred):
        kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
        recon_loss = 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
        return eta * recon_loss + alpha * kl_loss

    return kl_recon_loss


def kl_loss(mu, log_var, alpha=0.1):
    def kl_recon_loss(y_true, y_pred):
        kl_loss = 0.5 * K.mean(K.exp(log_var) + K.square(mu) - 1. - log_var, 1)
        return alpha * kl_loss

    return kl_recon_loss


def mmd(n_conditions, beta, kernel_method='multi-scale-rbf'):
    def mmd_loss(real_labels, y_pred):
        with tf.variable_scope("mmd_loss", reuse=tf.AUTO_REUSE):
            real_labels = K.reshape(K.cast(real_labels, 'int32'), (-1,))
            conditions_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=n_conditions)
            loss = 0.0
            for i in range(len(conditions_mmd)):
                for j in range(i):
                    loss += compute_mmd(conditions_mmd[j], conditions_mmd[j + 1], kernel_method)
            return beta * loss

    return mmd_loss


class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
            t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (
                    y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
            final = t1 + t2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)

        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.log(1.0 - self.pi + eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
            zero_case = -tf.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            ridge = self.ridge_lambda * tf.square(self.pi)
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = _nan2inf(result)

        return result


def nb_loss(disp, mu, log_var, scale_factor=1.0, alpha=0.1):
    kl = kl_loss(mu, log_var, alpha=alpha)

    def nb(y_true, y_pred):
        nb_obj = NB(theta=disp, masking=True, scale_factor=scale_factor)
        return nb_obj.loss(y_true, y_pred) + kl(y_true, y_pred)

    return nb


def zinb_loss(pi, disp, mu, log_var, ridge=0.1, alpha=0.1):
    kl = kl_loss(mu, log_var, alpha=alpha)

    def zinb(y_true, y_pred):
        zinb_obj = ZINB(pi, theta=disp, ridge_lambda=ridge)
        return zinb_obj.loss(y_true, y_pred) + kl(y_true, y_pred)

    return zinb


LOSSES = {
    "mse": kl_recon,
    "mmd": mmd,
    "nb": nb_loss,
    "zinb": zinb_loss,
    "cce": 'categorical_crossentropy',
}
