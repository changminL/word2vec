from __future__ import absolutclock deprecatede_import, division, print_function

import numpy as np
import multiprocessing
from multiprocessing import Pool, Array, Process, Value, Manager
import random
import os
import unicodedata
import time
from io import open

num_threads = multiprocessing.cpu_count()
start = time.process_time()
print(num_threads)
MAX_STRING = 100
NAX_SENTENCE_LENGTH = 1000
MAX_CODE_LENGTH = 40


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
	return ''.join(
			c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != 'Mn'
	)

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1",, s)
	s = re.sub(r"[a-zA-Z.!?]+", r" ", s)
	s = re.sub(r"\s+", r" ", s).strip()
	return s

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
		self.toal_words = 0

	def _init_dict(self, input_file, min_count):
		sentences = []
		for line in self.input_file:
			sentence = []
			line = line.strip().split(' ')

			for word in line:
				word = normalizeString(word)
				self.addWord(word)
				sentence.append[word]

			sentences.append(sentence)

		self.trim(min_count)

		for k, c in self.word2count.items():
			self.total_words += c

		return sentences

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
			self.index2count[self.word2index[word]] += 1

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

MIN_COUNT = 3
EPOCH = 5
WINDOW = 5
debug_mode = True
# Make a Skip-gram model
class SkipGram:
	def __init__(self, vocab, emb_dim):
		self.sentences = []
		self.vocab = vocab		
		self.embed_dim = emb_dim
		low = -0.5 / emb_dim
		high = 0.5 / emb_dim
		self.W = np.random.uniform(low, high, (self.vocab.num_words, emb_dim))
		self.W_prime = np.zeros((self.vocab.num_words, emb_dim))		


	def LoadData(self, tid):
		sentence_count = len(self.sentences)
		start = sentence_count // num_threads * t_id
		end = min(sentence_count // num_threads * (t_id + 1), sentence_count)
		return self.sentences[start:end]
		
	def TrainModelThread(self, tid):
		word_count = last_word_count = sentence_position = sentence_length = 0
		local_epochs = EPOCH
		sentences = self.LoadData(tid)
		
		neu1 = np.zeros(self.embed_dim)
		neu1e = np.zeros(self.embed_dim)
		sen = []
		for epoch in local_epochs:
			for sentence in sentences:

				if word_count - last_word_count > 10000:
					word_count_actual += word_count - last_word_count
					last_word_count = word_count
					if debug_mode:
						now = time.process_time()
						print("Learning rate: {:f}  Progress: {:.2f}  Words/thread/sec: {:.2f}k  ".format(
						lr, word_count_actual / (EPOCH * self.vocab.total_words + 1) * 100,
						word_count_actual / (now - start + 1) / 1e6 * 1000))

					lr = starting_lr * (1 - word_count_actual / (EPOCH * self.vocab.total_words + 1))
					if (lr < starting_lr * 0.0001): lr = starting_lr * 0.0001

				if sentence_length == 0:
					for word in sentence:
						# The subsampling randomly discards frequent words while keeping the ranking same
						if sample > 0:
							ran = (np.sqrt(self.vocab.word2count[word] / (sample * self.vocab.total_words)) + 1) *
							  	(sample * self.vocab.total_words) / self.vocab.word2count[word]
							if ran < np.random.uniform(0, 1, 1).item():
								continue
						sen.append(word)
						sentence_length += 1
					sentence_position = 0
							
				
				word = sen[sentence_position]
				
				neu1 = np.zeros(self.embed_dim)
				neu1e = np.zeros(self.embed_dim)
				b = np.random.randint(WINDOW, size=1).item()
				for 		
				sentence_position += 1
				if sentence_position >= sentence_length:
					sentence_length = 0
					continue
			word_count_actual += word_count - last_word_count
			word_count = 0
			last_word_count = 0
			sentence_length = 0
			
		
	def TrainModel(self, input_file_name):
		print("Starting training using file ", input_file_name)
		input_file = open(input_file_name, 'rb')
		# Initializing dictionary
		self.sentences = self.vocab._init_dict(input_file, MIN_COUNT)
		huffman = HuffmanTree(self.vocab)
		huffman.build_tree()
		start = time.process_time()
		jobs = []
		t_id = Value('i', 0)
		for i in range(num_threads):
			p = Process(target=self.TrainModelThread, args=[t_id])
			jobs.append(p)
			t_id = Value('i', t_id.value + 1)

		for j in jobs:
			j.start()

		for j in jobs:
			j.join()


		




skip = SkipGram(5, 2)
print(skip.W)
print(skip.W_prime)
