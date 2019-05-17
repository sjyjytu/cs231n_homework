import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
      fyi = X[i].dot(W[:,y[i]])
      loss -= fyi
      sum = 0.0
      for j in range(num_class):
          e_Wj_x = math.exp(X[i].dot(W[:,j]))
          sum += e_Wj_x
          dW[:,j] += e_Wj_x * X[i]
      loss += math.log(sum)
      dW /= sum
      dW[:, y[i]] -= X[i].T
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Li = -Wyi * Xi + log(sum(e^(Wj * Xi)))
  num_train = X.shape[0]
  num_class = W.shape[1]
  score = X.dot(W)
  score_correct = score[np.arange(num_train),y]
  sum_j_e_score = np.sum(np.exp(score),axis=1)
  loss += np.sum(np.log(sum_j_e_score) - score_correct) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  ef_di_sum = np.exp(score) / sum_j_e_score
  ef_di_sum[np.arange(num_train),y] -= 1
  dW = X.T.dot(ef_di_sum) / num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

