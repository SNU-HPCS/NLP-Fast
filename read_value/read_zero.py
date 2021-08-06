import numpy as np

batch = 2
head = 16
from_s = 256
to_s = 256
total = batch*head*from_s*to_s


def read_dump(path, file_name):
    readf = open(path+file_name, 'r')
    data = {}
    data_count = 0
    while(True):
        line = readf.readline()
        if not line:
            break
        if "attention_probs" in line:
            data_count += 1
            #attention_probs (b/n/f/t):(?/16/256/256) bert/encoder/layer_10/attention/self/Softmax:0[[[[
            name_index = line.find('layer')
            bracket_index = line.find('[')
            layer_num = line[name_index:name_index+8].replace("/", "")
            stripped_data = line[bracket_index:].replace("["," ").replace("]"," ")
            arr = np.fromstring(stripped_data, dtype='float32', sep=' ')
            if(len(arr) != total):
                print("read error")
            arr = arr.reshape(batch, head, from_s, to_s)
            data[layer_num]=arr
    
    #print(data_count)
    return data

def zero_count(data):
    total_zero={}
    zero_data={}
    skip_data={}
    for i in range(24):
        counter = np.zeros((10), dtype=int)
        layer_num = str('layer_'+ str(i))
        arr = data[layer_num]
        flat_arr = arr.flatten()
        
        if(len(flat_arr) != total):
            print("count error")
        
        for j in range(10):
            if(j==0):
                counter[j] = (arr == 0).sum()
                arr_skip = (1 - (arr == 0))
            else:
                counter[j] = (arr < (0.1)**j).sum()
                arr_skip = (1 - (arr < (0.1)**j))
            
            name = layer_num + '_th:' + str(j)
            zero_data[name] = counter[j]
            skip_data[name] = arr_skip

            if(i==0):
                total_zero[j] = counter[j]
            else: 
                total_zero[j] += counter[j]

    return [total_zero, zero_data, skip_data]


def column_count(skip_data):
    zero_col = {}
    total_zero_col = {}
    for k in range(10):
        for i in range(24):
            counter = np.zeros((10), dtype=int)
            name = 'layer_' + str(i) + '_th:' + str(k)
            arr = skip_data[name]
            zero_col_arr = (arr == 0).sum(2)
            counter[0] = (zero_col_arr==256).sum()
            counter[1] = (zero_col_arr >= int(256*0.98)).sum()
            counter[2] = (zero_col_arr >= int(256*0.96)).sum()
            counter[3] = (zero_col_arr >= int(256*0.94)).sum()
            counter[4] = (zero_col_arr >= int(256*0.92)).sum()
            counter[5] = (zero_col_arr >= int(256*0.90)).sum()
            counter[6] = (zero_col_arr >= int(256*0.88)).sum()
            counter[7] = (zero_col_arr >= int(256*0.86)).sum()
            counter[8] = (zero_col_arr >= int(256*0.84)).sum()
            counter[9] = (zero_col_arr >= int(256*0.82)).sum()

            for j in range(10):
                key = name + '_colskip%:' + str(j)
                zero_col[key] = counter[j]
                key_ig_layer = 'th:' + str(k) + '_colskip%:' + str(j)
                if(i==0):
                    total_zero_col[key_ig_layer] = counter[j]
                else:
                    total_zero_col[key_ig_layer] += counter[j]

    return [total_zero_col, zero_col]
        



def main():
    
    #data = read_dump('/home/joonsung/research/memNN/BERT/', 'training_attention_probs_fine_tuned.txt')
    data = read_dump('/home/joonsung/research/memNN/BERT/', 'training_attention_probs.txt')
    total_zero, zero_data, skip_data = zero_count(data)
    total_zero_col, zero_col = column_count(skip_data) 
    
    for i in range(10):
        for j in range(10):
            key = 'th:' + str(i) + '_colskip%:' + str(j)
            print(key + '_' + str(total_zero_col[key]))
   
    for i in range(10):
        print('th:' + str(i) + '_' + str(total_zero[i]))
    
    for i in range(10):
       for j in range(24):
           key = 'layer_' + str(j) + '_th:' + str(i)
           print(key + '_'+ str(zero_data[key]))


if __name__ == '__main__':
    main()               
