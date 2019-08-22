import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import torch.optim as optim
import os
import pdb

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

#Set your datafile path!
file_path = '/home/changmin/research/MMI/data/wikitext-2/wiki.train.tokens'

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(file_path)

class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.w_embeddings = nn.Embedding(2*emb_size-1, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(2*emb_size-1, emb_dimension, sparse=True)
        self._init_emb()

    def _init_emb(self):
        init_range = 0.5 / self.emb_dimension
        self.w_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_w, neg_v):
        emb_w = self.w_embeddings(pos_w)
        neg_emb_w = self.w_embeddings(neg_w)
        emb_v = self.v_embeddings(pos_v)
        neg_emb_v = self.v_embeddings(neg_v)

        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(-1 * score)

        neg_score = torch.mul(neg_emb_w, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(neg_score)

        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        loss = loss.to(device)
        return loss

    def save_embedding(self, id2word, file_name):
        embedding_v = self.v_embeddings.cpu()
        embedding_w = self.w_embeddings.cpu()
        embedding_v = embedding_v.weight.data.numpy()
        embedding_w = embedding_w.weight.data.numpy()
        embedding = (embedding_v + embedding_w) / 2
        fout = open(file_name, 'w')
        fout.write('%d %d\n' %(len(id2word), self.emb_dimension))
        for w_id, w in id2word.items():
            e = embedding[w_id]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

class HuffmanNode:
    def __init__(self, word_id, frequency):
        self.word_id = word_id
        self.frequency = frequency
        self.left_child = None
        self.right_child = None
        self.father = None
        self.Huffman_code = []
        self.path = []

class HuffmanTree:
    def __init__(self, word_id_frequency_dict):
        self.word_count = len(word_id_frequency_dict)
        self.word_id_code = dict()
        self.word_id_path = dict()
        self.root = None
        unmerge_node_list = [HuffmanNode(word_id, frequency) for word_id, frequency in
                            word_id_frequency_dict.items()]
        self.huffman = [HuffmanNode(word_id, frequency) for word_id, frequency in
                       word_id_frequency_dict.items()]

        self.build_tree(unmerge_node_list)

        self.generate_huffman_code_and_path()

    def merge_node(self, node1, node2):
        sum_frequency = node1.frequency + node2.frequency
        mid_node_id = len(self.huffman)
        father_node = HuffmanNode(mid_node_id, sum_frequency)
        if node1.frequency >= node2.frequency:
            father_node.left_child = node1
            father_node.right_child = node2
        else:
            father_node.left_child = node2
            father_node.right_child = node1
        self.huffman.append(father_node)
        return father_node

    def build_tree(self, node_list):
        while len(node_list) > 1:
            i1 = 0
            i2 = 1
            if node_list[i2].frequency < node_list[i1].frequency:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        [i1, i2] = [i2, i1]
            father_node = self.merge_node(node_list[i1], node_list[i2])
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, father_node)
        self.root = node_list[0]
		
    def generate_huffman_code_and_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()

            while node.left_child or node.right_child:
                code = node.Huffman_code
                path = node.path
                node.left_child.Huffman_code = code + [1]
                node.right_child.Huffman_code = code + [0]
                node.left_child.path = path + [node.word_id]
                node.right_child.path = path + [node.word_id]

                stack.append(node.right_child)
                node = node.left_child
            word_id = node.word_id
            word_code = node.Huffman_code
            word_path = node.path
            self.huffman[word_id].Huffman_code = word_code
            self.huffman[word_id].path = word_path

            self.word_id_code[word_id] = word_code
            self.word_id_path[word_id] = word_path

    def get_all_pos_and_neg_path(self):
        positive = []
        negative = []
        for word_id in range(self.word_count):
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.huffman[word_id].Huffman_code):
                if code == 1:
                    pos_id.append(self.huffman[word_id].path[i])
                else:
                    neg_id.append(self.huffman[word_id].path[i])
            positive.append(pos_id)
            negative.append(neg_id)
        return positive, negative

def test_huffmanTree():
    word_frequency = {0: 4, 1: 6, 2: 3, 3: 2, 4: 2}
    print(word_frequency)
    tree = HuffmanTree(word_frequency)
    print(tree.word_id_code)
    print(tree.word_id_path)
    for i in range(len(word_frequency)):
        print(tree.huffman[i].path)
    print(tree.get_all_pos_and_neg_path())


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
        self.num_words = 0 # Count default tokens

        for word in keep_words:
            self.addWord(word)

class InputData:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name)
        self.min_count = min_count
        self.Voc = Voc('skip_gram_HS')   
        self.word_count_sum = 0
        self.sentence_count = 0
        self._init_dict()
        self.huffman_tree = HuffmanTree(self.Voc.index2count)
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        
        print('Word Count is: ', self.Voc.num_words)
        print('Word Count Sum is:', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
        print('Tree Node is:', len(self.huffman_tree.huffman))
        
    def _init_dict(self):    
        for line in self.input_file:
            line = line.strip().split(' ')
            self.sentence_count += 1
            for word in line:
                self.Voc.addWord(word)
        
        self.Voc.trim(self.min_count)
        
        for k, c in self.Voc.word2count.items():
            self.word_count_sum += c
        
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
                    if word_idx_w == word_idx_v:  # Skip a center word
                        continue
                    result_pairs.append((word_idx_w, word_idx_v))
        return result_pairs
    
    def get_pairs(self, batch_pairs):
        neg_word_pair = []
        pos_word_pair = []
        for pair in batch_pairs:
            pos_word_pair += zip([pair[0]] * len(self.huffman_pos_path[pair[1]]), self.huffman_pos_path[pair[1]])
            neg_word_pair += zip([pair[0]] * len(self.huffman_neg_path[pair[1]]), self.huffman_neg_path[pair[1]])
        return pos_word_pair, neg_word_pair
    
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size

#hyper parameters
WINDOW_SIZE = 6
BATCH_SIZE = 64
MIN_COUNT = 3
EMB_DIMENSION = 300
LR = 0.02
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

        print("SkipGram Training......")
        all_pairs = self.data.get_all_pairs(WINDOW_SIZE)
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)


        print('Initializing')
        start_iteration = 1
        print_loss = 0
        n_iteration = ITERATION_NUM

        for iteration in range(start_iteration, n_iteration + 1):
            batch_pairs = [random.choice(all_pairs) for _ in range(BATCH_SIZE)]
            pos_pairs, neg_pairs = self.data.get_pairs(batch_pairs)
            pos_u = [int(pair[0]) for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_u = [int(pair[0]) for pair in neg_pairs]
            neg_v = [int(pair[1]) for pair in neg_pairs]

            pos_u = torch.LongTensor(pos_u).to(device)
            pos_v = torch.LongTensor(pos_v).to(device)
            neg_u = torch.LongTensor(neg_u).to(device)
            neg_v = torch.LongTensor(neg_v).to(device)

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_u, pos_v, neg_u, neg_v)
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

w2v = Word2Vec(input_file_name='/home/changmin/research/MMI/data/wiki2010/wiki2010.txt', output_file_name="word_embedding.txt")
print(w2v.data.Voc.word2count)
w2v.train()


"""
from sklearn.metrics.pairwise import cosine_similarity

f = open('word_embedding.txt')
f.readline()
all_embeddings = []
all_words = []
word2index = dict()
for i, line in enumerate(f):
    line = line.strip().split(' ')
    if len(line) == 100:
        word = ''
        embedding = [float(x) for x in line]
    else:
        word = line[0]
        embedding = [float(x) for x in line[1:]]
    assert len(embedding) == EMB_DIMENSION
    all_embeddings.append(embedding)
    all_words.append(word)
    word2index[word] = i
all_embeddings = np.array(all_embeddings)
while 1:
    word = input('Word: ')
    try:
        w_id = word2index[word]
    except:
        print('Cannot find this word')
        continue
    embedding = all_embeddings[w_id:w_id + 1]
    d = cosine_similarity(embedding, all_embeddings)[0]
    d = zip(all_words, d)
    d = sorted(d, key=lambda x : x[1], reverse=True)
    for w in d[:10]:
        if len(w[0]) < 2:
            continue
        print(w)
"""
