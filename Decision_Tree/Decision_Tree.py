import numpy as np

class Node:
    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.class_    = None
        self.left      = None
        self.right     = None