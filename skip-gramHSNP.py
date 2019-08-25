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

class Voc:
	def __init__(self):
		self.trimmed = False
		self.word2index = {}
		self.word2count = {}
		self.index2word = {}
		self.index2count = {}
		# For Huffman encoding
		self.index2code = {}
		self.index2point = {}
		self.index2codelen = {}
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
	def __init__(self, vocab):
		self.vocab = vocab
		self.vocab_size = len(self.vocab.index2count)

		self.count = np.ones(self.vocab_size * 2 + 1) * 1e15
		for word_id, frequency in self.vocab.index2count.items():
			self.count[word_id] = frequency

		self.binary = np.zeros(self.vocab_size * 2 + 1)
		self.parent = np.zeros(self.vocab_size * 2 + 1)

	def build_tree(self):
		min1_idx = min2_idx = int()
		pos1 = self.vocab_size - 1
		pos2 = self.vocab_size

		# Follwoing algorithm constructs the Huffman tree by adding one node at a time
		for i in range(self.vocab_size):
			# First, find two smallest nodes 'min1, min2'
			if pos1 >= 0:
				if self.count[pos1] < self.count[pos2]:
					min1_idx = pos1
					pos1 -= 1
				else:
					min1_idx = pos2
					pos2 += 1
			else:
				min1_idx = pos2
				pos2 += 1
			if pos1 >= 0:
				if self.count[pos1] < self.count[pos2]:
					min2_idx = pos1
					pos1 -= 1
				else:
					min2_idx = pos2
					pos2 += 1
			else:
				min2_idx = pos2
				pos2++
			self.count[self.vocab_size + i] = self.count[min1_idx] + self.count[min2_idx]
			self.parent[min1_idx] = self.vocab_size + i
			self.parent[min2_idx] = self.vocab_size + i
			self.binary[min2_idx] = 1
		
		# Now assign binary code to each vocabulary word
		for w_id in range(self.vocab_size):
			path_id = w_id
			code = np.array(list())
			point = np.array(list())
			while 1:
				np.insert(code, 0, binary[path_id])
				np.insert(point, 0, path_id)
				path_id = self.parent[path_id]
				if path_id == (self.vocab_size * 2 - 2):
					break
			point = point - self.vocab_size
			np.insert(point, 0, self.vocab_size - 2)
			self.vocab.index2codelen[w_id] = len(code)
			self.vocab.index2point[w_id] = point
			self.vocab.index2code[w_id] = code
			del code
			del point
		del self.count
		del self.binary
		del self.parent
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
