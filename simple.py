import numpy as np
import multiprocessing
from multiprocessing import Pool, Array, Process, Value, Manager
import random
import os
import time
import queue
from io import open

num_threads = multiprocessing.cpu_count()
start = 0
starting_lr = 1e-3
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

    def _init_dict(self, sentences, min_count):
        for line in sentences:
            self.add_sentence(line)
        
        self.trim(min_count)

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

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        q = queue.PriorityQueue()
        keep_words = 0
        for (k, v) in self.word2count.items():
            if v >= min_count:
                keep_words += v
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
            for _ in range(-freq):
                self.add_word(word)

MIN_COUNT = 3
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

    def forward(self, index1, index2):
	    start_idx1 = index1 * self.embed_dim 
        end_idx1 = (index1 + 1) * self.embed_dim
        start_idx2 = index2 * self.embed_dim
        end_idx2 = (index2 + 1) * self.embed_dim
        f = np.dot(W[start_idx1:end_idx1], W_prime[start_idx2:end_idx2])
        return f

    def backward(self, ff, label:
        if ff > MAX_EXP:
            gradient = (label - 1) 
        elif ff < -MAX_EXP:
            gradient = (label - 0) 
        else:
            gradient = (label - sigmoid(f))
        return gradient 

    def optimize(self, lr, gradients, W, W_prime):
        center_id = None
        center_grad_sum = np.zeros(self.embed_dim)
        for grad, center_i, context_i in gradients:
            center_id = center_i
            center_grad_sum += grad * np.array(W_prime[context_i*self.embed_dim:(context_i+1)*self.embed_dim])
            W_prime[context_i*self.embed_dim:(context_i+1)*self.embed_dim] += lr \ 
                                    * grad * np.array(W[center_i*self.embed_dim:(center_i+1)*self.embed_dim])
        W[center_id*self.embed_dim:(center_id+1)*self.embed_dim] += center_grad_sum
            

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
        
    def train_model_thread(self, tid, lr, word_count_actual, W, W_prime):
        word_count = 0
        sentence_count = len(self.sentences)

        start_idx = (sentence_count // num_threads) * tid
        end_idx = min((sentence_count // num_threads) * (tid + 1), sentence_count)

        sentences = self.sentences[start_idx:end_idx]

        for epoch in range(EPOCH):
            for sentence in sentences:
                line, w_cnt = self.subsampling(sample, sentence)
                word_count += w_cnt

                update_temp = np.zeros(self.embed_dim)
                line_pos = 0

                for word_idx in line:
                    soft_slide = np.random.randint(WINDOW, size=1).item()
                    for center_idx in line[max(line_pos - WINDOW + soft_slide, 0):line_pos + WINDOW + 1 - soft_slide]:

                           if self.vocab.index2count.get(center_idx) is None:
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
                               ff = self.forward(center_idx, context_idx)
                               gradients.append((self.backward(ff, label), center_idx, context_idx))

                           self.optimize(lr, gradients, W, W_prime)

                           
                    line_pos += 1                      
                                       
                 
        

        
    def train_model(self, input_file_name, output_file_name):
        print('Starting training using file ', input_file_name)
        input_file = open(input_file_name, 'r')
         
        self.sentences = input_file.readlines()

        self.vocab._init_dict(self.sentences, MIN_COUNT)

        self.init_unigram_table()

        word_count_actual = 0
        low = -0.5 / self.embed_dim
        high = 0.5 / self.embed_dim
        self.W = np.random.uniform(low, high, (self.vecab.num_words, self.embed_dim))
        self.W_prime = np.zeros((self.vocab.num_words, self.embed_dim))

        start = time.clock()
        jobs = []
 
        t_id = 0
        word_count_actual = Value('i', 0)
        lr = Value('d', 0.025)
        W = Array('d', self.W.reshape(-1))
        W_prime = Array('d', self.W_prime.reshape(-1))
        for i in range(num_threads):
            p = Process(target=self.train_model_thread, args=[t_id, lr,
                        word_count_actual, W, W_prime])
            jobs.append(p)
            t_id += 1

        for j in jobs:
            j.start()
 
        for j in jobs:
            j.join()

        self.W = np.array(W[:]).reshape(self.vocab.num_words, self.embed_dim)
        self.W_prime = np.array(W_prime[:]).reshape(self.vocab.num_words, self.embed_dim)
        self.save_embedding(output_file_name)

