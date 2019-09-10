# Word2vec

&nbsp;&nbsp;In this notebook, I'll lead you through using Numpy to implement the word2vec algorithm using the skip-gram architecture. By implementing this, you'll learn about embedding words for use in natural language processing. This will come in handy when dealing with things like machine translation.

## Readings (reference)
&nbsp;&nbsp;Here are the resources I used to build this notebook. I suggest reading these either beforehand or while your're working on this material.

[1] Francois Chaubard, Michael Fang, Guillaume Genthial, Rohit Mundra, Richard Socher, Winter 2019, CS224n: Natural Language Processing with Deep Learning 1 Lecture Notes: Part I Word Vectors I: Introduction, SVD and Word2Vec  
[2] Lilian Weng, (2017) 'Learning Word Embedding', *Github*, 15 Oct. Available at:https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html

### Iteration Based Methods - Word2vec

&nbsp;&nbsp;Let's take a look at a new approach. Instead of computing and storing global information about some huge dataset (which might be billions of sentences), we can try to create a model that will be able to learn one iteration at a time and eventually be able to encode the probability of a word given its context.
The idea is to design a model whose parameters are the word vectors. Then, train the model on a certain objective. At every iteration we run our model, evaluate the errors, and follow an update rule that has some notion of penalizing the model parameters that caused the error. Thus, we learn our word vectors. This idea is a very old one dating back to 1986. We call this method "backpropagating" the errors. The simpler the model and the task, the faster it will be to train it.

&nbsp;&nbsp;Several approaches have been tested design models for NLP whose first step is to transform each word in a vector. For each special task (Named Entity Recognition, Part-of-Speech tagging, etc.) they train not only the model's parameters but also the vectors and achieve great performance, while computing good word vectors!

&nbsp;&nbsp;In this class, we will present a simpler, more recent, probabilistic method by [Mikolov et al., 2013] : word2vec. Word2vec is a software package that actually includes:

- 2 **algorithms**: **continuous bag-of-words (CBOW)** and **skip-gram**. CBOW aims to predict a center word from the surrounding context in terms of word vectors. Skip-gram does the opposite, and predicts the *distribution* (probability) of context words from a center word.
- 2 **training methods**: **negative sampling** and **hierarchical softmax**. Negative sampling defines an objective by sampling *negative examples*, while hierarchical softmax defines an objective using an efficient tree structure to compute probabilities for all the vocabulary.

### Language Models(*Unigrams*, *Bigrams*, *etc*.)

&nbsp;&nbsp;First, we need to create such a model that will assign a probability to a sequence of tokens.     Let us start with an example:
<center>*"The cat jumped over the puddle."*</center>

&nbsp;&nbsp;A good language model will give this sentence a high probability because this is a completely valid sentence, syntactically and semantically. Similary, the sentence "stock boil fish is toy" should have a very low probability because it makes no sense. Mathematically, we can call this probability on any given sequence of *n* words:
$$P(w_{1},w_{2},\cdots,w_{n}) $$
![equation](https://latex.codecogs.com/gif.latex?P%28w_%7B1%7D%2Cw_%7B2%7D%2C%5Ccdots%2Cw_%7Bn%7D%29)
&nbsp;&nbsp;We can take the unary language model approach and break apart this probability by assuming the word occurences are completely independent:
$$P(w_{1},w_{2},\cdots,w_{n}) = \prod_{i=1}^{n}P(w_{i})$$
&nbsp;&nbsp;However, we know this is a bit ludicrous because we know the next word is highly contingent upon the previous sequence of words. And the silly sentence example might actually score highly. So perhaps we let the probability of the sequence depend on the pairwise probability of a word in the sequence and the word next to it. We call this the bigram model and represent it as:
$$P(w_{1},w_{2},\cdots,w_{n})=\prod_{i=2}^{n}P(w_{i}|w_{i-1})$$
&nbsp;&nbsp;Again this is certainly a bit naive since we are only concerning ourselves with pairs of neighboring words rather than evaluating a whole sentence, but as we will see, this representation gets us pretty far long. Note in the Word-Word Matrix with context of size 1, we basically can learn these pairwise probabilities. But again, this would require computing and storing global information about a massive dataset.

&nbsp;&nbsp;Now that we understand how we can think about a sequence of tokens having a probability, let us observe some example models that could learn these probabilities

### Continuous Bag of Words Model (CBOW)

&nbsp;&nbsp;One approach is to treat {"The", "cat", "over", "the", "puddle"} as a context and from these words, be able to predict or generate the center word "jumped". This type of model we call a Continuous Bag of Words (CBOW) Model.

&nbsp;&nbsp;Let's discuss the CBOW Model above in greater detail. First, we set up our known parameters. Let the known parameters in our model be the sentence represented by one-hot word vectors. The input one hot vectors or context we will represent with an $x^{(c)}$. And the output as $y^{(c)}$ and in the CBOW model, since we only have one output, so we just call this $y$ which is the one hot vector of the known center word. Now, let's define our unknowns in our model.

&nbsp;&nbsp;We create two matrices, $W \in \mathbb{R}^{n\times|V|}$ and $W' \in \mathbb{R}^{|V|\times n}$. Where $n$ is an arbitrary size which defines the size of our embedding space. $W$ is the input word matrix such that the *i*-th column of $W$ is the n-dimensional embedded vector for word $w_{i}$ when it is an input to this model. We denote this $n \times 1$ vector as $v_{i}$. Similary, $W{'}$ is the output word matrix. The *j*-th row of $W'$ is an *n*-dimensional embedded vector for word $w_{j}$ when it is an output of the model. We denote this row of $W{'}$ as $w{'}_{j}$. Note that we do in fact learn two vectors for every word $word_{i}$ (i.e. input word vector $w_{i}$ and output word vector $w'_{i}$).

![CBOW](./image/CBOW.png)
[image source from Lilian Weng's blog](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html "Lilian Weng")

&nbsp;&nbsp;We breakdown the way this model works in these steps:
1. We generate our one hot word vectors for the input context of size
$m:(x^{(c-m)},...,x^{(c-1)},x^{(c+1)},...,x{(c+m)} \in \mathbb{R}^{|V|})$.
2. We get our embedded word vectors for the context $(w_{c-m} = Wx^{(c-m)},w_{c-m+1} = Wx^{(c-m+1)},...,w_{c+m} = Wx^{(c+m)} \in \mathbb{R}^{n})$
3. Average these vectors to get $\hat{w} = \frac{w_{c-m}+w_{c-m+1}+...+w_{c+m}}{2m} \in \mathbb{R}^{n}$ 
4. Generate a score vector $z = W{'}\hat{w} \in \mathbb{R}^{|V|}$. As the dot product of similar vectors is higher, it will push similar words close to each other in order to achieve a high score.
5. Turn the scores into probabilities $\hat{y}=softmax(z) \in \mathbb{R}^{|V|}$
6. We desire our probabilities generated, $\hat{y} \in \mathbb{R}^{|V|}$, to match the true probabilities,$y \in \mathbb{R}^{|V|}$ , which also happens to be the one hot vector of the actual word.

&nbsp;&nbsp;So now that we have an understanding of how our model would work if we had a W and W', how would we learn these two matrices? Well, we need to create an objective function. Very often when we are trying to learn a probability from some true probability, we look to information theory to give us a measure of the distance between two distributions. Here, we use a popular choice of distance/loss measure, corss entropy $H(\hat{y},y)$.

&nbsp;&nbsp;The intuition for the use of cross-entropy in the discrete case can be derived from the formulation of the loss funciton:
$$H(\hat{y},y) = - \sum_{j=1}^{|V|}y_{i}\log(\hat{y_{j}})$$

&nbsp;&nbsp;Let us concern ourselves with the case at hand, which is that $y$ is a one-hot vector. Thus we know that the above loss simplifies to simply:
$$H(\hat{y},y) = - y_{i}log(\hat{y_{j}})$$

&nbsp;&nbsp;In this formulation, $c$ is the index where the correct word's one hot vector is 1. We can now consider the case where our prediction was perfect and thus $\hat{y_{c}} = 1$. We can then calculate $H(\hat{y},y) = -1log(1) = 0$. Thus,for a perfect prediction, we face no penalty or loss. Now let us consider the opposite case where our prediction was very bad and thus $\hat{y_{c}} = 0.01$. As before, we can calculate our loss to be $H(\hat{y},y) = -1log(0.01) \approx 4.605$. We can thus see that for probability distributions, cross entropy provides us with a good measure of distance. We thus formulate our optimization objective as: 

$minimize J  = -\log P(word_{c}|word_{c-m},...,word_{c-1},word_{c+1},...,word_{c+m}) \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\;= -\log P(w'_{c}|\hat{w}) \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\; = -\log \frac{exp(w{'}_{c}^{T}\hat{w})}{\sum_{j=1}^{|V|} exp(w{'}_{j}^{T}\hat{w}) } \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\; = -w{'}_{c}^{T} \hat{w} + \log \sum_{j=1}^{|V|} exp(w{'}_{j}^{T} \hat{w}) $

&nbsp;&nbsp;We use stochastic gradient descent to update all relevant word vectors $w{'}_{c}$ and $w_{j}$.

### Skip-Gram Model
&nbsp;&nbsp;Another approach to create a model such that given the center word "jumped", the model will be able to predict or generate the surrounding words "The", "cat", "over", "the", "puddle". Here we call the word "jumped" the context. We call this type of model a Skip-Gram model.

![Skip-gram](./image/SkipGram.png)
[image source from Lilian Weng's blog](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html "Lilian Weng")

&nbsp;&nbsp;Let's discuss the Skip-Gram model above. The setup is largely the same but we essentially swap our $x$ and $y$ i.e. $x$ in the CBOW are now $y$ and vice-versa. The input one hot vector (center word) we will represent with an $x$ (since there is only one). And the output vectors as $y^{(j)}$. We define $W$ and $W{'}$ the same as in CBOW.

&nbsp;&nbsp;We breakdown the way this model works in these 6 steps:
1. We generate our one hot input vector $x \in \mathbb{R}^{|V|}$ of the center word
2. We get our embedded word vector for the center word $w_{c} = Wx \in \mathbb{R}^{n}$
3. Generate a score vector $z = W{'}w_{c}$
4. Turn the score vector into probabilities, $\hat{y} = softmax(z)$. Note that $\hat{y}_{c-m},...,\hat{y}_{c-1},\hat{y}_{c+1},...,\hat{y}_{c+m}$ are the probabilities of observing each context word.
5. We desire our probability vector generated to match the true probabilities which is $y^{(c-m)},...,y^{(c-1)},y^{(c+1)},...,y^{(c+m)}$, the one hot vectors of the actual output.

&nbsp;&nbsp;As in CBOW, we need to generate an objective function for us to evaluate the model. A key difference here is that we invoke a Naive Bayes assumption to break out the probabilities. If you have not seen this before, then simply put, it is a strong (naive) conditional independence assumption. In other words, given the center word, all output words are completely independent.

$minimize J  = -\log P(word_{c-m},...,word_{c-1},word_{c+1},...,word_{c+m}|word_{c}) \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\;= -\log \prod_{j=0,j \neq m}^{2m} P(word_{c-m+j}|word_{c}) \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\; = -\log \prod_{j=0,j \neq m}^{2m} P(w{'}_{c-m+j}|w_{c}) \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\; = -\log \prod_{j=0,j \neq m}^{2m} \frac{exp(w{'}_{c-m+j}^{T}w_{c})}{\sum_{k=1}^{|V|}exp(w{'}_{k}^{T}w_{c})} \\ \;\;\;\;\;\;\;\;\;\;\;\;\;\; = - \sum_{j=0, j \neq m}^{2m} w{'}_{c-m+j}^{T}w_{c} + 2m\log \sum_{k=1}^{|V|} exp(w{'}_{k}^{T}w_{c})$

&nbsp;&nbsp;With this objective function, we can compute the gradients with respect to the unknown parameters and at each iteration update them via Stochastic Gradient Descent.

&nbsp;&nbsp;Note that
$$J = - \sum_{j=0,j \neq m}^{2m} \log P(w{'}_{c-m+j}|w_{c})$$
$$= \sum_{j=0, j \neq m}^{2m} H(\hat{y}, y_{c-m+j})$$

&nbsp;&nbsp;where $H(\hat{y},y_{c-m+j})$ is the cross-entropy between the probability vector $\hat{y}$ and one-hot vector $y_{c-m+j}$ 

### Negative Sampling

Lets take a second to look at objective function. Note that the summation over $|V|$ is computationally huge! Any update we do or evaluation of the objective funciton would take $O(|V|)$ time which if we recall is in the millions. A simple idea is we could instead just approximate it.

&nbsp;&nbsp;For every training step, instead of looping over the entire vocabulary, we can just sample several negative samples! We "sample" from a noise distribution $(P_{n}(w))$ whose probabilities match the ordering of the frequency of the vocabulary. To augment our formulation of the problem to incorporate Negative Sampling, all we need to do is update the:
* objective function
* gradients
* update rules

[**MIKOLOV ET AL.**](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) present **Negative Sampling** in DISTRIBUTED REPRESENTATIONS OF WORDS AND PHRASES AND THEIR COMPOSITIONALITY. While negative sampling is based on Skip-Gram model, it is infact optimizing a different objective. Consider a pair $(w,c)$ of word and context. Did this pair come from the training data? Let's denote by $P(D=1|w,c)$ the probability that $(w,c)$ came from the corpus data. Correspondingly, $P(D=0|w,c)$ will be the probability that $(w,c)$ did not come from the corpus data. First, let's model $P(D=1|w,c)$ with the sigmoid function:
$$P(D=1|w,c,\theta) = \sigma(w_{c}^{T}w_{w}) = \frac{1}{1 + e^{(-w_{c}^{T}w_{w})}}$$

&nbsp;&nbsp;Now, we build a new objective function that tries to maximize the probability of a word and context being in the corpus data if it indeed is, and maximize the probability of a word and context not being in the corpus data if it indeed is not. We take a simple maximum likelihood approach of these two probabilities. (Here we take $\theta$ to be the parameters of the model, and in our case it is $W$ and $W{'}$.)

$\theta = argmax_{\theta} \prod_{(w,c) \in D} P(D = 1|w,c,\theta) \prod_{(w,c) \in \tilde{D}} P(D = 0|w,c,\theta) \\ \;\,= argmax_{\theta} \prod_{(w,c) \in D} P(D = 1|w,c,\theta) \prod_{(w,c) \in \tilde{D}} (1 - P(D = 1|w,c,\theta)) \\ \;\, = argmax_{\theta} \sum_{(w,c) \in D} \log P(D = 1|w,c,\theta) + \sum_{(w,c) \in \tilde{D}} \log(1 - P(D=1|w,c,\theta)) \\ \;\, = argmax_{\theta} \sum_{(w,c) \in D} \log \frac{1}{1 + exp(-w{'}^{T}_{w}w_{c})} + \sum_{(w,c) \in \tilde{D}} \log(1- \frac{1}{1+exp(-w{'}^{T}_{w}w_{c})}) \\ \;\, = argmax_{\theta} \sum_{(w,c) \in D} \log \frac{1}{1 + exp(-w{'}^{T}_{w}w_{c})} + \sum_{(w,c) \in \tilde{D}} \log( \frac{1}{1+exp(w{'}^{T}_{w}w_{c})})$

Note that maximizing the likelihood is the same as minimizing the negative log likelihood
$$ J = - \sum_{(w,c) \in D} \log \frac{1}{1 + exp(-w{'}^{T}_{w}w_{c})} - \sum_{(w,c) \in \tilde{D}} \log( \frac{1}{1 + exp(w{'}_{w}^{T}w_{c})} )$$

Note that $\tilde{D}$ is a "false" or "negative" corpus. Where we would have sentences like "stock boil fish is toy". Unnatural sentences that should get a low probability of ever occurring. We can generate $\tilde{D}$ on the fly by randomly sampling this negative form the word bank.

For skip-gram, our new objective function for observing the context word c-m+j given the center word c would be
$$ -\log \sigma(w{'}^{T}_{c-m+j} \cdot w_{c}) - \sum_{k=1}^{K} \log \sigma(-w{'}^{T}_{k} \cdot w_{c})$$


For CBOW, our new objective function for observing the center word $w{'}_{c}$ given the context vector $\tilde{w} = \frac{w_{c-m}+w_{c-m+1}+...+w_{c+m}}{2m}$ would be 

$$ -\log \sigma(w{'}_{c}^{T} \cdot \tilde{w}) - \sum_{k=1}^{K} \log \sigma(-\tilde{w{'}}^{T}_{k} \cdot \tilde{w})$$

In the above formulation, $\{ \tilde{u}_{K}|k=1...K \}$ are sampled from $P_{n}(w)$. Let's discuss what $P_{n}(w)$ should be. While there is much discussion of what makes the best approximation, what seems to work best is the Unigram Model raised to the power of 3/4. Why 3/4? Here's an example that might help gain some intuition:

$$is: 0.9^{3/4} = 0.92$$
$$Constitution: 0.09^{3/4} = 0.16$$
$$bombastic: 0.01^{3/4} = 0.032$$

"Bombastic" is now 3x more likely to be sampled while "is" only went up marginally.
