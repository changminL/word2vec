import numpy as np
import multiprocessing
from multiprocessing import Pool, Array, Process, Value, Manager
import random
import os
import time
from io import open

num_threads = multiprocessing.cpu_count()
start = time.process_time()
starting_lr = 1e-3
sample = 1e-5
word_count_actual = 0
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

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

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

        for (k, v) in self.word2count.items():
            if v >= min_count:
                for _ in range(v):
                    keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), \
               len(self.word2index), len(keep_words)
               / len(self.word2index)))

        # Reinitialize dictionaires
       
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0

        for word in keep_words:
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

    def LoadData(self, tid):
        sentence_count = len(self.sentences)
        start = sentence_count // num_threads * tid
        end = min(sentence_count // num_threads * (tid + 1),
                  sentence_count)
        return self.sentences[start:end]

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
                neu1 = np.zeros(self.embed_dim)
                neu1e = np.zeros(self.embed_dim)

                b = np.random.randint(WINDOW, size=1).item()
                
