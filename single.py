import numpy as np
import multiprocessing
from multiprocessing import Pool, Array, Process, Value, Manager
import random
import os
import time
import queue
from io import open
import pdb

num_threads = multiprocessing.cpu_count()
start = 0
starting_lr = 0.025
sample = 1e-5
word_count_actual = 0
negative = 15
lr = 0.025

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0
        self.total_words = 0

    def _init_dict(self, sentences, min_count, max_count):
        for line in sentences:
            self.add_sentence(line)
        
        self.trim(min_count, max_count)

        for (k, c) in self.word2count.items():
            self.total_words += c

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
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

    def trim(self, min_count, max_count):
        if self.trimmed:
            return
        self.trimmed = True
        q = queue.PriorityQueue()
        keep_words = 0
        for (k, v) in self.word2count.items():
            if v >= min_count:
                keep_words += 1
                q.put((v, k))

        print('keep_words {} / {} = {:.4f}'.format(keep_words, \
              len(self.word2index), keep_words
              / len(self.word2index)))

        # Reinitialize dictionaries

        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0
     
        while not q.empty():
            freq, word = q.get()
            if freq < max_count:
                for _ in range(freq):
                    self.add_word(word)

MIN_COUNT = 3
MAX_COUNT = 100000
MAX_EXP = 6
EPOCH = 3
WINDOW = 6
debug_mode = True

class SkipGram:

    def __init__(self, vocab, embedding_dimension):
        self.sentences = []
        self.vocab = vocab
        self.embed_dim = embedding_dimension
        self.W = None
        self.W_prime = None
        self.table = []

    def init_unigram_table(self):
        table_size = 1e8
        pow_frequency = np.array(list(self.vocab.index2count.values())) ** 0.75
        word_pow_sum = np.sum(pow_frequency)
        ratio_array = pow_frequency / word_pow_sum
        word_count_list = np.around(ratio_array * table_size)
        for word_index, word_freq in enumerate(word_count_list):
            self.table += [word_index] * int(word_freq)
        self.table = np.array(self.table)

    def save_embedding(self, file_name):
        embedding = self.W
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(self.vocab.index2word), self.embed_dim))
        for (w_id, w) in self.vocab.index2word.items():
            e = embedding[w_id]
            e = " ",join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

    def forward(self, index1, index2, W, W_prime):
        f = np.dot(W[index1], W_prime[index2])
        return f

    def backward(self, ff, label):
        if ff > MAX_EXP:
            gradient = (label - 1) 
        elif ff < -MAX_EXP:
            gradient = (label - 0) 
        else:
            gradient = (label - sigmoid(ff))
        return gradient 

    def optimize(self, lr, gradients, W, W_prime):
        center_id = None
        center_grad_sum = np.zeros(self.embed_dim)
        for grad, center_i, context_i in gradients:
            center_id = center_i
            center_grad_sum += grad * np.array(W_prime[context_i])
            W_prime[context_i] += lr * grad * np.array(W[center_i])
        W[center_id] += center_grad_sum
            
    def subsampling(self, sample_bound, sentence):
        w_cnt = 0
        ret = []
        line = sentence.strip().split(' ')
        for word in line:
            if self.vocab.word2count.get(word) == None:
                continue
            if sample_bound > 0:
                ran = (np.sqrt(self.vocab.word2count[word]
                      / (sample * self.vocab.total_words)) + 1) \
                      * (sample * self.vocab.total_words) \
                      / self.vocab.word2count[word]
                if ran < np.random.random():
                    continue
            ret.append(self.vocab.word2index[word])
            w_cnt += 1

        return ret, w_cnt
                             
    def train(self, input_file_name, output_file_name, lr):
        print('Starting training using file ', input_file_name)
        pdb.set_trace()
        input_file = open(input_file_name, 'r')
        # Read sentences from a input file
        self.sentences = input_file.readlines()
        # Initialize a vocabulary with a training corpus
        self.vocab._init_dict(self.sentences, MIN_COUNT, MAX_COUNT)
        # Also construct the unigram language model
        self.init_unigram_table()

        # Initialize weights
        word_count_actual = 0
        low = -0.5 / self.embed_dim
        high = 0.5 / self.embed_dim
        self.W = np.random.uniform(low, high, (self.vocab.num_words, self.embed_dim))
        self.W_prime = np.zeros((self.vocab.num_words, self.embed_dim))
        
        w_count = prev_w_count = 0

        for epoch in range(EPOCH):
            for sentence in self.sentences:
                line, w_cnt = self.subsampling(sample, sentence)
                w_count += w_cnt

                line_pos = 0

                for word_idx in line:
                    soft_slide = np.random.randint(WINDOW, size=1).item()
                    for center_idx in line[max(line_pos - WINDOW + soft_slide,0):line_pos + WINDOW + 1 - soft_slide]:
                        if self.vocab.index2count.get(center_idx) is None:
                            continue
                        if center_idx == word_idx:
                            continue
                        center_idx = int(center_idx)

                        gradients = []
                        for neg_sample in range(negative+1):
                            if neg_sample == 0:
                                context_idx = word_idx
                                label = 1
                            else:
                                rand = np.random.randint(int(len(self.table)), size=1).item()
                                context_idx = int(self.table[rand])
                                if context_idx == 0:
                                    context_idx = np.random.randint(self.vocab.num_words, size=1).item()
                                if context_idx == word_idx:
                                    continue
                                label = 0
                            ff = self.forward(center_idx, context_idx, self.W, self.W_prime)
                            gradients.append((self.backward(ff, label), center_idx, context_idx))

                        self.optimize(lr, gradients, self.W, self.W_prime)

                    line_pos += 1

        self.save_embedding(output_file_name)

input_file_name = '/home/changmin/project/word2vec/wiki_0001.txt'
output_file_name = 'embedding_results.txt'
voc = Voc()
skip = SkipGram(voc, 100)
skip.train(input_file_name, output_file_name, 0.025)
