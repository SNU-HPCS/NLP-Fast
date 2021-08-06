import tensorflow as tf
import numpy as np
import os 

sess=tf.Session()
new=tf.train.import_meta_graph('/home/suyeon/output_final/model.ckpt-130319.meta')
model=new.restore(sess, tf.train.latest_checkpoint('/home/suyeon/output_final/'))
tvars=tf.trainable_variables()
tvars_vals=sess.run(tvars)


for var, val in zip(tvars, tvars_vals):
    name = var.name
    name = name.replace("/", ":")
    fn = '/home/suyeon/research/weight/'+str(name)
    f = open(fn, 'w')
    #np.savetxt(f, val.shape, fmt="%d")
    np.savetxt(f, val, fmt="%f")
    f.close()
