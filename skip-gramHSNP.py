from __future__ import absolute_import, division, print_function

import numpy as np
import multiprocessing
from multiprocessing import Pool, Array, Process, Value, Manager
import random
import os
import unicodedata
from io import open

num_threads = multiprocessing.cpu_count()
print(num_threads)
MAX_STRING 100
NAX_SENTENCE_LENGTH 1000
MAX_CODE_LENGTH 40


class Voc_word:
	def __init__(self):
		self.count = None
		self.parent = None
		self.word = None
		self.code = None
		self.code_length = None

class Voc:
	def __init__(self):
		self.trimmed = False
		self.word2index = {}
		self.word2count = {}
		self.index2word = {}
		self.index2count = {}
		self.num_words = 0

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.index2count[self.num_words] = 1
			self.num_words += 1
		else:
			self.word2count[word] += 1
			self.index2count[self.num_words] = 1

	# Remove words below a certain count threshold
	def trim(self, min_count):
		if self.trimmed:
			return
		self.trimmed = True

		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				for _ in range(v):
					keep_words.append(k)

		print('keep_words {} / {} = {:.4f}'.format(
			len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
		))

		# Reinitialize dictionaries
		self.word2index = {}
		self.word2count = {}
		self.index2word = {}
		self.index2count = {}
		self.num_words = 0

		for word in keep_words:
			self.addWord(word)

class HuffmanTree:
	def __init__(self, word_id_frequency_dict):
		self.vocab_size = len(word_id_frequency_dict)

		self.count_list = np.ones(self.vocab_size * 2 + 1) * 1e15
		for word_id, frequency in word_id_frequency_dict.items():
			self.count_list[word_id] = frequency

		self.binary_list = np.zeros(self.vocab_size * 2 + 1)

	def build_tree(self):
		pos1 = self.vocab_size - 1
		pos2 = self.vocab_size

		# Follwoing algorithm constructs the Huffman tree by adding one node at a time
		for i in range(self.vocab_size):
			# First, 

# Make a Skip-gram model
class SkipGram:
	def __init__(self, vocab_size, emb_dim):
		low = -0.5 / emb_dim
		high = 0.5 / emb_dim
		self.W = np.random.uniform(low, high, (vocab_size, emb_dim))
		self.W_prime = np.zeros((vocab_size, emb_dim))

		self.createBinaryTree()

def TrainModelThread(tid, sentences):






skip = SkipGram(5, 2)
print(skip.W)
print(skip.W_prime)
