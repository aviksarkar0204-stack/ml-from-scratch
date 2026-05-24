import numpy as np

class Node:
    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.class_    = None
        self.left      = None
        self.right     = None

def gini(labels):
    n = len(labels)
    if n == 0:
        return 0
    classes = np.unique(labels)
    impurity = 1.0
    for c in classes:
        p = np.sum(labels == c) / n
        impurity -= p * p
    return impurity

def best_split(X, y):
    best_gini   = 1e9
    best_feat   = None
    best_thresh = None

    for f in range(X.shape[1]):
        for thresh in np.unique(X[:, f]):
            left_mask  = X[:, f] < thresh
            right_mask = ~left_mask

            g = (np.sum(left_mask)/len(y))  * gini(y[left_mask]) \
              + (np.sum(right_mask)/len(y)) * gini(y[right_mask])

            if g < best_gini:
                best_gini   = g
                best_feat   = f
                best_thresh = thresh

    return best_feat, best_thresh

def build_tree(X, y, depth=0, max_depth=3):
    node = Node()

    # leaf conditions
    if depth >= max_depth or len(np.unique(y)) == 1:
        node.class_ = int(np.bincount(y).argmax())
        return node

    # find best split
    feat, thresh = best_split(X, y)

    # split data
    left_mask  = X[:, feat] < thresh
    right_mask = ~left_mask

    # set node values
    node.feature   = feat
    node.threshold = thresh

    # recurse
    node.left  = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth)
    node.right = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)

    return node