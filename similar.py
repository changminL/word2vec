from sklearn.metrics.pairwise import cosine_similarity

EMB_DIMENSION = 100

f = open()
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
