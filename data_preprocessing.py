from __future__ import absolute_import
from __future__ import division
from __future__ import division

import os
# nltk for nlp processing
import nltk

# way for serializing way
import pickle
import random


pad_Token = 0
go_Token = 1
eos_Token = 2
unknown_Token = 3

class Batch:
    ''' the batch class, to record encoder, decoder, decoder tag and decoder length
    '''
    def __init__(self):
         self.encoder_Sequence = []
         self.decoder_Sequence = []
         self.target_Sequence = []
         self.weights = []

def loadData(filename):
    ''' read the sample data
    parameter filename: filepath
    return: word2id, id2word,trainingSamples
    '''
    data_path = os.path.join(filename)
    print("loading data from {}".format(data_path))
    with open(data_path,'rb') as handle:
    	data = pickle.load(handle)
    	word_to_id = data['word2id']
    	id_to_word = data['id2word']
    	trainingSamples = data['trainingSamples']
    return word_to_id,id_to_word,trainingSamples


def initializeBatch(samples,en_de_seq_len):
	'''
	according to given samples, using padding to make placeholder form data
	parameter samples: one batch, every line is form of Q and A
	parameter en_de_seq_len: length of encoder and length of decoder
	return: form of feed_dick
	'''
	batch = Batch()
	batch_size = len(samples)
	for i in range(batch_size):
		sample = samples[i]

		# input reversed order, can improve the model performance
		batch.encoder_Sequence.append(list(reversed(sample[0])))
		# add the go and eos Token
		batch.decoder_Sequence.append([go_Token]+sample[1]+[eos_Token])
		# ignore the go Token
		batch.target_Sequence.append(batch.decoder_Sequence[-1][1:])

		# pad every element to the fxied length
		batch.encoder_Sequence[i] = [pad_Token] * (en_de_seq_len[0] - len(batch.encoder_Sequence[i])) + batch.encoder_Sequence[i]
		batch.weights.append([1.0] * len(batch.target_Sequence[i]) + [0.0] * (en_de_seq_len[1] - len(batch.target_Sequence[i])))
		batch.decoder_Sequence[i] = batch.decoder_Sequence[i] + [pad_Token] * (en_de_seq_len[1] - len(batch.decoder_Sequence[i]))
		batch.target_Sequence[i] = batch.target_Sequence[i] + [pad_Token] * (en_de_seq_len[1] - len(batch.target_Sequence[i]))
	encoderSeqsT = []  # Corrected orientation
	for i in range(en_de_seq_len[0]):
		encoderSeqT = []
		for j in range(batch_size):
			encoderSeqT.append(batch.encoder_Sequence[j][i])
		encoderSeqsT.append(encoderSeqT)
	batch.encoder_Sequence = encoderSeqsT
	decoder_Sequence_T = []
	target_Sequence_T = []
	weights_T = []
	for i in range(en_de_seq_len[1]):
		decoder_SequenceT = []
		target_SequenceT = []
		weightsT = []
		for j in range(batch_size):
			decoder_SequenceT.append(batch.decoder_Sequence[j][i])
			target_SequenceT.append(batch.target_Sequence[j][i])
			weightsT.append(batch.weights[j][i])
		decoder_Sequence_T.append(decoder_SequenceT)
		target_Sequence_T.append(target_SequenceT)
		weights_T.append(weightsT)
	batch.decoder_Sequence = decoder_Sequence_T
	batch.target_Sequence = target_Sequence_T
	batch.weights = weights_T
	return batch




def getBatch(data,batch_size,en_de_seq_len):
	'''
	according to the batch_size, divide the original data into small batch
	and use initializeBatch to initialize data.
	parameter data: Q and A
	parameter batch_size: clear
	parameter en_de_seq_len:
	return : data to feed_dict
	'''

	# firs to shuffle
	random.shuffle(data)

	# the set of batch
	batches = []
	data_length = len(data)
	def genNextSamples():
		for i in range(0,data_length,batch_size):
			# generate a batch of data to feed the tensorflow network
			yield data[i:min(i+batch_size,data_length)]

	for samples in genNextSamples():
		batch = initializeBatch(samples,en_de_seq_len)
		batches.append(batch)

	# return the processed batches of data
	return batches


def sentence_to_encoder(sentence,word2id,en_de_seq_len):
	'''
	transfer the sentence into data which can be feed into model.
	parameter sentence: the input sentence
	parameter word2id: the dictionary
	parameter en_de_seq_len
	'''
	if sentence == '':
		return None
	# split the word
	tokens = nltk.word_tokenize(sentence)
	if len(tokens) > en_de_seq_len[0]:
		return None
	# transfer every word into id
	wordIds = []
	for token in tokens:
		wordIds.append(word2id.get(token, unknown_Token))
	# initialzie the Batch
	batch = initializeBatch([[wordIds, []]], en_de_seq_len)
	return batch
	





