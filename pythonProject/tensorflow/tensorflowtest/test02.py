import tensorflow as tf

matrixl = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrixl,matrix2)

#启动默认 图
sess = tf.Session()

# result = sess.run(product)
# print (result)
# sess.close()
with tf.Session() as sess:
    result = sess.run(product)
    print(result)