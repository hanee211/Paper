import tensorflow as tf

sample1 = " if you want you"
sample2 = " if if you you y"
idx2char = list(set(sample1))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idx

sample_idx1 = [char2idx[c] for c in sample1]  # char to index
sample_idx2 = [char2idx[c] for c in sample2]  # char to index
x_data = [sample_idx1[:-1], sample_idx2[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx1[1:], sample_idx2[1:]]   # Y label sample (1 ~ n)  hello: ello

dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 2 # one sample data, one batch
sequence_length = len(sample1) - 1  # number of lstm unfolding (unit #)
hidden_size = rnn_hidden_size
decoder_num_symbols = num_classes
decode_seq_length = sequence_length
decoder_embedding_size = 3


X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

#cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_hidden_size)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

print(cell)
W_decoder = tf.Variable(tf.random_normal([hidden_size, decoder_num_symbols], stddev=0.35))
B_decoder = tf.Variable(tf.zeros([decoder_num_symbols]))
decoder_output_projection = (W_decoder,B_decoder)


Input = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(decode_seq_length)]
Target = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(decode_seq_length)]
encode_last_state = tf.placeholder(dtype=tf.float32, shape=[None, hidden_size])
feed_Previous_Value = tf.placeholder(tf.bool)




decode_outputs, decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(Input, encode_last_state, cell, decoder_num_symbols, decoder_embedding_size, output_projection=None, feed_previous=True, update_embedding_for_previous=True, scope=None)
'''
decoder_symbols = [tf.matmul(_decode_output, W_decoder) for _decode_output in decode_outputs]

loss_weights = [ tf.ones_like(y, dtype=tf.float32) for y in Input ]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_symbols, Target, loss_weights)

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
predict = tf.argmax(decoder_symbols, axis=2)
'''




'''
OUTPUTS = tf.placeholder(tf.float32, [None, sequence_length, )

batch_size x sequence_length x num_decoder_symbols

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=OUTPUTS, targets=Y,weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)


with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   for i in range(3000):
       l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
       result = sess.run(prediction, feed_dict={X: x_data})
       # print char using dic
       result_str = [idx2char[c] for c in np.squeeze(result)]
       print(i, "loss:", l, "Prediction:", ''.join(result_str))
'''