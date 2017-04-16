import tensorflow as tf
import numpy as np
import datetime as dt


hidden_size = 20
batch_size = 0
#================================================================
#================================================================
# Set Environment

word_file_name = 'word'
word_embedding_file_name = 'word_embeddings'



#================================================================
#================================================================

#-----------------------------------------
# For decoder
decoder_num_symbols = 50002
decoder_embedding_size = 20

PAD = 0
#-----------------------------------------


def indexing():
	word_file = open(word_file_name, 'r')
	embedding_file = open(word_embedding_file_name, 'r')

	idx2word = list()
	idx2w_embedding = list()
	
	
	for w in word_file:
		idx2word.append(w.replace('\n', ''))
	idx2word.append(PAD)
	idx2word.append("go_decoder")

	word2idx = {w:i for i, w in enumerate(idx2word)}
	
	
	for l in embedding_file:
		idx2w_embedding.append([float(num) for num in l.replace('\n', '').split(' ')])

	idx2w_embedding = np.array(idx2w_embedding)


	add_vector = [[0 for i in range(128)]]
	idx2w_embedding = np.append(idx2w_embedding, add_vector, axis=0)

	return idx2word, word2idx, idx2w_embedding


	
def encoding_text(text, idx2word, word2idx, idx2w_embedding):
	max_len = 0
	encoding_text = list()
	embedding_text = list()
	
	for sentence in text:
		sentence = sentence.replace('\n', '').lower().split(' ')
		encoding_text.append([word2idx[w] for w in sentence if w in word2idx])
		embedding_text.append([idx2w_embedding[word2idx[w]] for w in sentence if w in word2idx])
		
		if(max_len < len(sentence)):
			max_len = len(sentence)
			
	for sentence, em_sentence in zip(encoding_text, embedding_text):
		if len(sentence) < max_len:
			pad_size = max_len - len(sentence)
			sentence.extend([PAD for i in range(pad_size)])
			em_sentence.extend([idx2w_embedding[word2idx[PAD]] for i in range(pad_size)])
			
	return encoding_text, embedding_text;

	
	
#--------------------------------------------------------------------------	
#--------------------------------------------------------------------------	
#
#							 Start Program 
#
#--------------------------------------------------------------------------	
#--------------------------------------------------------------------------	

text = ["I like this city and this country very much",
		"I have great time with my family today"]

'''"My room has a lot of books and one table and one chair and bookshelf",
		"I am living in Korea",
		"asfsadfasd asdfasdfasd asfdasfasd I adfadsf",
		"This is a great book I have ever read in my life",
		"This is incredible cookies and I think no one can easily write them"]'''
#text = ["I like this city and this country very much"]

idx2word, word2idx, idx2w_embedding = indexing()
e_text, em_text = encoding_text(text, idx2word, word2idx, idx2w_embedding)


'''target_e_text = list()
for s in e_text:
	t_s = [0] + s[:-1]
	target_e_text.append(t_s)'''

# For Encoder
seq_length = len(e_text[0])	
batch_size = len(e_text)
encoder_input_dim = len(em_text[0][0])


e_text_tran = np.transpose(e_text)
em_text_tran = np.transpose(em_text)

decoder_e_text = e_text
for i in range(batch_size):
	decoder_e_text[i] =  [word2idx["go_decoder"]] + decoder_e_text[i][:-1]
decoder_e_text_tran = np.transpose(decoder_e_text)
	
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#print(seq_length)
#print(batch_size)
#print(encoder_input_dim)
#print(em_text[0])

# For decoder
decode_seq_length = seq_length


			
#_______________________________________________________________________________
#   Encoder Part
#_______________________________________________________________________________
			

cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
encoder_Input = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, encoder_input_dim])

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, encoder_Input, initial_state=initial_state, dtype=tf.float32)

#_______________________________________________________________________________
#   Decoder Part
#_______________________________________________________________________________

W_decoder = tf.Variable(tf.random_normal([hidden_size, decoder_num_symbols], stddev=0.35))
B_decoder = tf.Variable(tf.zeros([decoder_num_symbols]))
decoder_output_projection = (W_decoder,B_decoder)

Input = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(decode_seq_length)]
Target = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(decode_seq_length)]
encode_last_state = tf.placeholder(dtype=tf.float32, shape=[None, hidden_size])
feed_Previous_Value = tf.placeholder(tf.bool)


decode_outputs, decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(Input, encode_last_state, cell, decoder_num_symbols, decoder_embedding_size, output_projection=None, feed_previous=True, update_embedding_for_previous=True, scope=None)
decoder_symbols = [tf.matmul(_decode_output, W_decoder) for _decode_output in decode_outputs]

loss_weights = [ tf.ones_like(y, dtype=tf.float32) for y in Input ]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoder_symbols, Target, loss_weights)

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
predict = tf.argmax(decoder_symbols, axis=2)


#_______________________________________________________________________________
#   Decoder Part
#_______________________________________________________________________________

feed_Encoder = dict()
feed_Decoder_Input = dict()
feed_Test = dict()
feed_Decoder_target = dict()
feed_Decoder = dict()
feed_Encoder[encoder_Input] = em_text
#feed_data[feed_Previous_Value] = tf.constant(False)



for i in range(seq_length):
	feed_Decoder_Input[Input[i]] = decoder_e_text_tran[i]
	feed_Decoder_target[Target[i]] = e_text_tran[i]
	feed_Test[Input[i]] = [word2idx["go_decoder"] for j in range(len(decoder_e_text_tran[i]))]
	
feed_Decoder = feed_Decoder_Input
feed_Decoder.update(feed_Decoder_target)

'''
#intput_data = 
#
decode input data 만들어야 하고,
decode input/target 을 만들고
input 에서 state 생성 후 decode 상태로 넣을 때, 

그리고 encode 생성을 초기화 할 수 있도록..
'''

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

print("###########################################################")
start_time = dt.datetime.now()

for i in range(3000):
	#feed_data[feed_Previous_Value] = tf.constant(False)

	#Encoder RUN
	res_outputs, res_states = sess.run([outputs, _states], feed_dict=feed_Encoder)
	
	#Decoder RUN
	feed_Decoder[encode_last_state] = res_states 
	feed_Decoder_Input[encode_last_state] = res_states 
	feed_Test[encode_last_state] = res_states 
	
	l, _ = sess.run([loss,train], feed_dict=feed_Decoder)
	
	if i % 100 == 0:
		#feed_data[feed_Previous_Value] = tf.constant(True)
		
		result = sess.run(predict,  feed_dict=feed_Test)
		#result = sess.run(predict,  feed_dict=feed_Decoder_Input)
		result = np.array(result)
		result = np.transpose(result)
		print("=============", i ,"==============================")
		for s in result:
			#snt = [idx2word[idx] for idx in s if idx != 1 ]
			snt = [idx2word[idx] for idx in s if idx2word[idx] != "go_decoder" and idx2word[idx] != 'UNK']
			snt = ' '.join(snt)
			print(snt)
		print("__________________________________________________")
		print("Take", str((dt.datetime.now() - start_time).seconds), "seconds for 100 cycles")
		start_time = dt.datetime.now()

	