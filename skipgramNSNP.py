import numpy as np
import multiprocessing
from multiprocessing import Pool, Array, Process, Value, Manager
import random
import os
import time
import queue
from io import open

num_threads = multiprocessing.cpu_count()
start = time.process_time()
starting_lr = 1e-3
sample = 1e-5
table_size = 1e8
word_count_actual = 0
negative = 5
lr = 0.025
print(num_threads)
MAX_STRING = 100
MAX_SENTENCE_LENGTH = 10
MAX_CODE_LENGTH = 40

def sigmoid(x, derivative=False):
    sigm = 1.  (1. + np.exp(-x))
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

        # For Huffman encoding

        self.index2code = {}
        self.index2point = {}
        self.index2codelen = {}
        self.num_words = 0
        self.total_words = 0

    def _init_dict(self, input_file, min_count):
        # Customize for text8 data

        sentences = []
        line = input_file.read()
        line = line.strip().split(" ")
        for word in line:
            self.addWord(word)
            sentences.append(word)
        self.trim(min_count)

        for (k, c) in self.word2count.items():
            self.total_words += c

        return sentences

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.index2count[self.num_words] 1
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
   
        for (k, v) in self.word2count.items():
            if v >= min_count:
                q.put((-v, k))

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), \
               len(self.word2index), len(keep_words)
               / len(self.word2index)))

        # Reinitialize dictionaires
       
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0

        while not q.empty():
            freq, word = q.get()
            for _ in range(-freq):
                self.addWord(word)

MIN_COUNT = 3
MAX_EXP = 6
EPOCH = 2
WINDOW = 5
debug_mode = True

# Make a Skip-gram model

class SkipGram:

    def __init__(self, vocab, emb_dim):
        self.sentences = []
        self.vocab = vocab
        self.embed_dim = emb_dim
        self.W = None
        self.W_prime = None
        self.table = None

    def InitUnigramTable(self):
        train_words_pow = 0
        d1 = power = 0.75
        self.table = np.zeros(table_size)
        for count in self.vocab.index2count.values():
            train_words_pow += np.power(count, power)
        i = 0
        d1 = np.power(self.vocab.index2count[i], power) / train_words_pow
        for a in range(table_size):
            self.table[a] = i
            if (a / table_size) > d1:
                i += 1
                d1 += pow(self.vocab.index2count[i], power) / train_words_pow
            if (i >= self.vocab.num_words):
                i = self.vocab.num_words - 1

    def SaveEmbedding(self, file_name):
        embedding = self.W
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(self.vocab.index2word),
                   self.embed_dim))
        for (w_id, w) in self.vocab.index2word.items():
            e = embedding[w_id]
            e = " ".join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

    def TrainModelThread(self, tid, lr, word_count_actual, W, W_prime):
        word_count = last_word_count = sentence_position = \
        sentence_length = 0

        local_epochs = EPOCH
        # Load Data for each threads
        sentence_count = len(self.sentences)

        start = (sentence_count // num_threads) * tid
        end = min((sentence_count // num_threads) * (tid + 1), sentence_count)

        sentences = self.sentences[start:end]

        neu1 = np.zeros(self.embed_dim)
        neu1e = np.zeros(self.embed_dim)
        sen = []
        eof = False
        for epoch in range(local_epochs):
            print("Start epoch: ", epoch)
            word_pos = 0
            while 1:
                if word_count - last_word_count > 10000:
                    word_count_actual.value = word_count_actual.value + word_count \
                                              - last_word_count
                    last_word_count = word_count
                    if debug_mode:
                        now = time.process_time()
                        print('Learning rate: {:f}   Progress: {:.2f}   Words/thread/sec: {:.2f}k   '.format(lr.value,
                            word_count_actual.value / (EPOCH
                            * self.vocab.total_words + 1)
                            * 100, word_count_actual.value
                            / (now - start + 1) ))
                    lr.value = starting_lr * (1 - word_count_actual.value / (EPOCH
                            * self.vocab.total_words + 1))
                    if lr.value < starting_lr * 0.0001:
                        lr.value = starting_lr * 0.0001

                if sentence_length == 0:
                    while 1:
                        if word_pos == len(sentences):
                            eof = True
                            break
                        word = sentences[word_pos]
                        word_pos += 1
                        word_count += 1
                        if self.vocab.word2count.get(word) == None:
                            continue
                        if sample > 0:
                            ran = (np.sqrt(self.vocab.word2count[word]
                                  / (sample * self.vocab.total_words)) + 1) \
                                  * (sample * self.vocab.total_words) \
                                  / self.vocab.word2count[word]
                            if ran < np.random.uniform(0, 1, 1).item():
                                continue
                        sen.append(self.vocab.word2index[word])
                        sentence_length += 1
                        if sentence_length >= MAX_SENTENCE_LENGTH:
                            break
                    sentence_position = 0

                if eof:
                    word_count_actual.value = word_count_actual.value + word_count - last_word_count
                    word_count = 0
                    last_word_count = 0
                    sentence_length = 0
                    word_pos = 0
                    break
                word_idx = sen[sentence_position]
                
                neu1e = np.zeros(self.embed_dim)

                b = np.random.randint(WINDOW, size=1).item()
                for a in range(b, WINDOW * 2 + 1 - b, 1):
                    if a != WINDOW:
                        last_pos = sentence_position - WINDOW + a
                        if last_pos < 0:
                            continue
                        if last_pos >= sentence_length:
                            continue

                        last_word_idx = sen[last_pos]
                        if self.vocab.index2count.get(last_word_idx) == None:
                            continue

                        l1 = int(last_word_idx)
                        neu1e = np.zeros(self.embed_dim)

                        # NEGATIVE SAMPLING
                        for d in range(negative+1):
                            if d == 0:
                                target = word_idx
                                label = 1
                            else:
                                target = np.random.randint(table_size, size=1).item()
                                if target == 0:
                                    target = np.random.randint(WINDOW
                                if target == word_idx:
                                    continue        
                                label = 0        
                            l2 = target 
                            f = 0
                            f += np.dot(W[l1*self.embed_dim:(l1+1)*self.embed_dim], W_prime[l2*self.embed_dim:(l2+1)*self.embed_dim])
                            if f > MAX_EXP:
                                gradient = (label - 1) * lr.value
                            elif f < -MAX_EXP:
                                gradient = (label - 0) * lr.value
                            else:
                                gradient = (label - sigmoid(f)) * lr.value
                            neu1e += gradient * np.array(W_prime[l2*self.embed_dim:(l2+1)*self.embed_dim])

                            W_prime[l2*self.embed_dim:(l2+1)*self.embed_dim] += gradient \
                                                                                * np.array(W[l1*self.embed_dim:(l1+1)*self.embed_dim])
                        W[l1*self.embed_dim:(l1+1)*self.embed_dim] += neu1e
                sentence_position += 1

                if sentence_position >= sentence_length:
                    sentence_length = 0
                    continue

    def TrainModel(self, input_file_name, output_file_name):
        print('Starting training using file ', input_file_name)
        input_file = open(input_file_name, 'r')
        
        # Initializing dictionary
         
        self.sentences = self.vocab._init_dict(input_file, MIN_COUNT)
        
        self.InitUnigramTable()

        word_count_actual = 0
        low = -0.5 / self.embed_dim
        high = -.5 / self.embed_dim
        self.W = np.random.uniform(low, high, (self.vocab.num_words, self.embed_dim))
        self.W_prime = np.zeros((self.vocab.num_words, self.embed_dim))

        start = time.process_time()
        jobs = []
        word_count_actual = Value('i', 0)
        lr = Value('d', 0.025)
        W = Array('d', self.W.reshape(-1))
        W_prime = Array('d', self.W_prime.reshape(-1))
        for i in range(num_threads):
            p = Process(target=self.TrainModelThread, args=[t_id, lr,
                        word_count_actual, W, W_prime])
            jobs.append(p)
            t_id += 1

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

         self.W = np.array(W[:]).reshape(self.vocab.num_words, self.embed_dim)
         self.W_prime = np.array(W_prime[:]).reshape(self.vocab.num_words, self.embed_dim)
         self.SaveEmbedding(output_file_name)

input_file_name = '/home/changmin/research/MMI/data/text8'
output_file_name = 'embeddingNS.txt'
read_file = open(input_file_name, 'r')
voc = Voc()
skip = SkipGram(voc, 100)
skip.TrainModel(input_file_name, output_file_name)

