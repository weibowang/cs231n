import numpy as np
from random import shuffle
from past.builtins import xrange

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
  #pass
  #http://blog.csdn.net/zt_1995/article/details/62227603
  #https://zhuanlan.zhihu.com/p/21485970
  for i in range (0, X.shape[0]):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scores = np.exp(scores)
    correct_class_score = scores[y[i]]
    scores_sum = np.sum(scores)
    p = lambda k: scores[k] / scores_sum
    loss += -np.log(correct_class_score / scores_sum)
    for j in range (0, dW.shape[1]):
      if (j != y[i]):
        dW[:, j] += p(j) * X[i]
      else:
        dW[:, j] += (p(j) - 1) * X[i]
        
  
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg*W
      
  return loss, dW
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
  #pass
  scores = X.dot(W)
  scores -= scores.max(axis = 1).reshape(scores.shape[0], 1)
  scores = np.exp(scores)  #500 * 10
  correct_class_scores = scores[np.arange(scores.shape[0]), y]
  scores_sum = scores.sum(axis=1) # (500,)
  loss += -np.sum(np.log(correct_class_scores / scores_sum))
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)

  scores_sum = scores_sum.reshape(scores_sum.shape[0], 1)  #(500, 1)
  scores = scores / scores_sum  #500 * 10
  scores[np.arange(scores.shape[0]), y] -= 1
  dW = X.T.dot(scores)
  dW /= X.shape[0]
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

