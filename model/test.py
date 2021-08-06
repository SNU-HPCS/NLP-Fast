import tensorflow as tf
import numpy as np
import math

def read_weight(s, r, c):
    textf = open(s, 'r')
    read = textf.read().split()
    weight = []
    
    for item in read:
        weight.append(float(item))
    
    weight = np.array(weight)
    weight = tf.convert_to_tensor(weight)
    print(weight.shape)
    weight = tf.reshape(weight, [r, c])
    weight = tf.cast(weight, dtype=tf.float32)
    return weight
def read_2d_input(s, b, r):
    textf = open(s, 'r')
    read = textf.read().split()
    result = []

    for item in read:
        result.append(float(item))
    result = np.array(result)
    print(result.shape)
    result = tf.convert_to_tensor(result)
    result = tf.reshape(result, [b,r])
    return result
    
def read_input(s, b, r, c):
    textf = open(s, 'r')
    read = textf.read().split()
    result = []

    for item in read:
        result.append(float(item))
    result = np.array(result)
    print(result.shape)
    result = tf.convert_to_tensor(result)
    result = tf.reshape(result, [b,r,c])
    result = tf.cast(result, dtype=tf.float32)
    return result

def read_4d_input(s, b1, b2, r, c):
    textf = open(s, 'r')
    read = textf.read().split()
    result = []

    for item in read:
        result.append(float(item))
    
    result = np.array(result)
    print(result.shape)
    result = tf.convert_to_tensor(result)
    result = tf.reshape(result, [b1, b2,r,c])
    return result

def transpose_for_score(input_tensor, batch_size, num_attention_heads,seq_length, width):
    output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
    result  = tf.transpose(output_tensor, [0, 2, 1, 3])
    return result

def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])

def FC(a, weight, bias):
    if(a==None): return
    output = tf.matmul(a, weight) + bias
    return output
def create_attention_mask(mask, batch, seq):
    ones = tf.ones(shape=[batch, seq, 1], dtype=tf.float32)
    to_mask = tf.reshape(mask, [batch, 1, seq])
    result = ones * to_mask
    write_t(result, './smalltest/attention__mask', False, 3);
    return result

def attention(a, attention_mask, qlw, qlb, klw, klb, vlw, vlb, batch, num_heads, seq, head_size):
 
    a = reshape_to_matrix(a)
    query = FC(a, qlw, qlb)
    write_t(query, './smalltest/query2d', False, 2)
    key = FC(a, klw, klb)
    write_t(key, './smalltest/key2d', False, 2)
    value = FC(a, vlw, vlb)
    write_t(value, './smalltest/value2d', False, 2)
    query = transpose_for_score(query, batch, num_heads, seq, head_size)
    write_t(query, './smalltest/query', False, 4)
    
    key = transpose_for_score(key, batch, num_heads, seq, head_size)
    write_t(key, './smalltest/key', False, 4)


    attention_score = tf.matmul(query, key, transpose_b =True)
    write_t(attention_score, './smalltest/attention_score', False, 4)

    attention_score = tf.multiply(attention_score, 1.0/math.sqrt(float(head_size)))
    write_t(attention_score, './smalltest/attention_norm_score', False, 4)
    

    if attention_mask is not None:
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        adder = (1.0 - tf.cast(attention_mask, tf.float32))*-10000.0
        attention_score += adder
    
    write_t(attention_score, './smalltest/attention_masking_score', False, 4)
    
    attention_probs = tf.nn.softmax(attention_score)
    write_t(attention_probs, './smalltest/attention_probs', False, 4)

    value = tf.reshape(value, [batch, seq, num_heads, head_size])
    value = tf.transpose(value, [0, 2, 1, 3])
    context = tf.matmul(attention_probs, value)

    write_t(context, './smalltest/4d_context', False, 4)
    write_t(value, './smalltest/value', False, 4)
    
    context = tf.transpose(context, [0, 2, 1, 3])
    context = tf.reshape(context, [batch*seq, num_heads*head_size])
    write_t(context, './smalltest/context', False, 2) 
    return context


def print_t(x):
    sess = tf.Session()
    x = tf.Print(x, [x], summarize=100000000)
    xshape = tf.shape(x)
    #print(sess.run(xshape))
    print(sess.run(x))
    
def write_t(x, s, is_int, rank):
    sess = tf.Session()
    var = sess.run(x)
    print(s)
    print(var.shape)
    f=open(s, 'w')
    if(rank==2):
        if(is_int):
            np.savetxt(f, var, fmt="%d")
        else:
            np.savetxt(f, var, fmt="%f")
    
    if(rank==3):
        for i in range(var.shape[0]):    
            np.savetxt(f, var[i], fmt="%f")
    
    if(rank==4):
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                np.savetxt(f, var[i][j], fmt="%f")
    
    
    f.close()
    
def diff(a, b, b1, b2, r, c):
    diff = 0
    for i in range(b1):
        for j in range(b2):
            for k in range(r):
                 for l in range(c):
                     diff += (a[i][j][k][l]-b[i][j][k][l])*(a[i][j][k][l]-b[i][j][k][l])
    print('@@@@@@@@@@@@@@@@@@@@@@@@diff@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(diff)

def pooler(kernel, bias, sequence_output):
    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(first_token_tensor.shape)
    #write_t(first_token_tensor, './first_token_tensor')

    pooled_output = tf.matmul(first_token_tensor, kernel) + bias
    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(pooled_output.shape)
    #write_t(pooled_output, './pooled_output')

def embedding(embedding_table, input_ids, vocabsize):
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocabsize)
    output = tf.matmul(one_hot_input_ids, embedding_table)
    #write_t(output, './smalltest/embedding_lookup', False)
    

def create_weight(vocab_size, hidden_size, batch, seq):
    #input_ids = np.random.randint(vocab_size, size=(batch, seq))
    #input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
     
    #embedding_table = np.random.rand(vocab_size, hidden_size)
    #embedding_table = tf.convert_to_tensor(embedding_table)
    
    input_mask = np.array([[1,1,1,1,1,1,1,0],[1,1,1,1,1,1,0,0]])
    input_mask = tf.convert_to_tensor(input_mask)
    write_t(input_mask, './smalltest/input_mask', False, 2)
     
    attention_input = np.random.rand(batch, seq, hidden_size)
    attention_input = tf.convert_to_tensor(attention_input)
    write_t(attention_input, './smalltest/attention_input', False, 3)

    qlw = np.random.rand(hidden_size, hidden_size)
    qlb = np.random.rand(hidden_size)
    qlw  = tf.convert_to_tensor(qlw)
    qlb = tf.convert_to_tensor(qlb)
    write_t(qlw, './smalltest/qlw', False, 2)
    write_t(qlb, './smalltest/qlb', False, 2)

    klw = np.random.rand(hidden_size, hidden_size)
    klb = np.random.rand(hidden_size)
    klw  = tf.convert_to_tensor(klw)
    klb = tf.convert_to_tensor(klb)
    write_t(klw, './smalltest/klw', False, 2)
    write_t(klb, './smalltest/klb', False, 2)
     
    
    vlw = np.random.rand(hidden_size, hidden_size)
    vlb = np.random.rand(hidden_size)
    vlw  = tf.convert_to_tensor(vlw)
    vlb = tf.convert_to_tensor(vlb)
    write_t(vlw, './smalltest/vlw', False, 2)
    write_t(vlb, './smalltest/vlb', False, 2)


def main():
    #vocab=10, hidden=6, batch=2, seq=8, #head=3
    '''
    create_weight(10, 6, 2, 8)
    embedding_table = read_weight("./smalltest/embedding_table", 10, 6)
    input_ids = read_2d_input("./smalltest/input_ids", 4, 8)
    input_ids = tf.cast(input_ids, dtype=tf.int32)
    embedding(embedding_table, input_ids, 10)
    
    
    qlw = read_weight("./smalltest/qlw", 6, 6)
    klw = read_weight("./smalltest/klw", 6, 6)
    vlw = read_weight("./smalltest/vlw", 6, 6)
    qlb = read_weight("./smalltest/qlb", 1, 6)
    klb = read_weight("./smalltest/klb", 1, 6)
    vlb = read_weight("./smalltest/vlb", 1, 6)
    '''
    #x = read_input("./smalltest/attention_input", 2, 8, 6)
    mask = read_weight("./smalltest/input_mask", 2, 8)
    attention_mask = create_attention_mask(mask, 2, 8)
    #attention(x, attention_mask, qlw, qlb, klw, klb, vlw, vlb, 2, 3, 8, 2)
   
    '''
    pweight = read_weight("../weight/bert:pooler:dense:kernel:0", 1024, 1024)
    sequence_output = read_input("../smallset/self.sequence_output394.txt", 8, 256, 1024)
    pooler(pweight, pbias, sequence_output)    
    '''

if __name__ == '__main__':
    main()
