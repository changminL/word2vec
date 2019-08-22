import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
import os

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.w_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self._init_emb()

    def _init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.w_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_v):
        emb_w = self.w_embeddings(pos_w)
        emb_v = self.v_embeddings(pos_v)
        neg_emb_v = self.v_embeddings(neg_v)

        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_score = torch.bmm(neg_emb_v, emb_v.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        loss = loss.to(device)
        return loss

    def save_embedding(self, idx2word, file_name):
        embedding_v = self.v_embeddings.cpu()
        embedding_w = self.w_embeddings.cpu()
        embedding_v = embedding_v.weight.data.numpy()
        embedding_w = embedding_w.weight.data.numpy()
        embedding = (embedding_v + embedding_w) / 2
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(idx2word), self.emb_dimension))
        for wid, w in idx2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

class Voc:
    def __init__(self, name):
        self.name = name
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
                    
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))
        
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2count = {}
        self.num_words = 0
        
        for word in keep_words:
            self.addWord(word)

class InputData:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name)
        self.min_count = min_count
        self.Voc = Voc('skip_gram_NS')
        self.word_count_sum = 0
        self.sentence_count = 0
        self._init_dict()
        self.sample_table = []
        self._init_unigram_table()
        
    def _init_dict(self):
        for line in self.input_file:
            line = line.strip().split(' ')
            self.sentence_count += 1
            for word in line:
                self.Voc.addWord(word)
                
        self.Voc.trim(self.min_count)
        
        for k, c in self.Voc.word2count.items():
            self.word_count_sum += c  
    
    def _init_unigram_table(self):
        table_size = 1e8
        pow_frequency = np.array(list(self.Voc.index2count.values())) ** 0.75
        word_pow_sum = sum(pow_frequency)
        ratio_array = pow_frequency / word_pow_sum
        word_count_list = np.round(ratio_array * table_size)
        for word_index, word_freq in enumerate(word_count_list):
            self.sample_table += [word_index] * int(word_freq)
        self.sample_table = np.array(self.sample_table)
        
    def get_all_pairs(self, window_size):
        result_pairs = []
        self.input_file = open(self.input_file_name, encoding="utf-8")
        sentences = self.input_file.readlines()
        for sentence in sentences:
            if sentence is None or sentence == '':
                continue
            word2idx_list = []
            for word in sentence.strip().split(' '):
                try:
                    word_index = self.Voc.word2index[word]
                    word2idx_list.append(word_index)
                except:
                    continue

            for i, word_idx_w in enumerate(word2idx_list):
                for j, word_idx_v in enumerate(word2idx_list[max(i - window_size, 0):i + window_size + 1]):
                    assert word_idx_w < self.Voc.num_words
                    assert word_idx_v < self.Voc.num_words
                    if word_idx_w == word_idx_v:
                        continue
                    result_pairs.append((word_idx_w, word_idx_v))
        return result_pairs
    
    def get_negative_sampling(self, positive_pairs, neg_count):
        neg_v = np.random.choice(self.sample_table, size=(len(positive_pairs), neg_count)).tolist()
        return neg_v
    
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size


test_data = ''

WINDOW_SIZE = 6
BATCH_SIZE = 64
MIN_COUNT = 3
EMB_DIMENSION = 300
LR = 0.02
NEG_COUNT = 5
ITERATION_NUM = 10000000
save_dir = './'
save_every = 500000
print_every = 100
decay_every = 500000

class Word2Vec:
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = InputData(input_file_name, MIN_COUNT)
        self.model = SkipGramModel(self.data.Voc.num_words, EMB_DIMENSION)
        self.model = self.model.to(device)
        self.model.train()
        self.lr = LR
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        # If you have cuda, configure cuda to call
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def train(self):
        print('SkipGram Training......')
        all_pairs = self.data.get_all_pairs(WINDOW_SIZE)
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count")
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)

        print('Initializing')
        start_iteration = 1
        print_loss = 0
        n_iteration = ITERATION_NUM

        for iteration in range(start_iteration, n_iteration + 1):
            batch_pairs = [random.choice(all_pairs) for _ in range(BATCH_SIZE)]
            pos_w = [int(pair[0]) for pair in batch_pairs]
            pos_v = [int(pair[1]) for pair in batch_pairs]
            neg_v = self.data.get_negative_sampling(batch_pairs, NEG_COUNT)

            pos_w = torch.LongTensor(pos_w).to(device)
            pos_v = torch.LongTensor(pos_v).to(device)
            neg_v = torch.LongTensor(neg_v).to(device)

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_w, pos_v, neg_v)
            print_loss += (loss.item() / BATCH_SIZE)
            loss.backward()
            self.optimizer.step()

            if iteration % decay_every == 0:
                self.lr = self.lr * 0.9
                print("The learning rate is changed to {:.6f}.".format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(save_dir, self.data.Voc.name, '{}-{}_{}'.format(EMB_DIMENSION, WINDOW_SIZE, BATCH_SIZE))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'model': self.model.state_dict(),
                    'opt': self.optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.data.Voc.__dict__
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

        self.model.save_embedding(self.data.Voc.index2word, self.output_file_name)


w2v = Word2Vec(input_file_name='/home/changmin/research/MMI/data/wiki2010/wiki2010.txt', output_file_name="word_embedding2.txt")
w2v.train()

