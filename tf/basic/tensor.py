#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os,sys
import tensorflow as tf
import numpy as np

#e.g. 1
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)
print result


