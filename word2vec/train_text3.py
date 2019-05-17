#-*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
path = os.getcwd()
size = 100
import random

with open(path[:-9] + '\\aclImdb\\train\\pos_all.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open(path[:-9] + '\\aclImdb\\train\\neg_all.txt', 'r') as infile:
    neg_tweets = infile.readlines()

with open(path[:-9] + '\\aclImdb\\train\\unsup_all.txt', 'r') as infile:
    unsup_reviews = infile.readlines()

# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)

# 零星的预处理
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('n', '') for z in corpus]
    corpus = [z.replace('&lt;br /&gt;', ' ') for z in corpus]

    # 将标点视为一个单词
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)


# Gensim 的 Doc2Vec 工具要求每个文档/段落包含一个与之关联的标签。我们利用 LabeledSentence 进行处理。格式形如 “TRAIN_i” 或者 “TEST_i”，其中 “i” 是假的评论索引。
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')


# 实例化 DM 和 DBOW 模型
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

a = []
a.extend(x_train)
a.extend(x_test)
a.extend(unsup_reviews)
# 对所有评论创建词汇表
model_dm.build_vocab(a)
model_dbow.build_vocab(a)

# 多次传入数据集，通过每次滑动（shuffling）来提高准确率。
all_train_reviews = np.concatenate((x_train, unsup_reviews))
b =[]
b.extend(x_train)
b.extend(unsup_reviews)

def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return (shuffled)


for epoch in range(1):
    model_dm.train(sentences_perm(b), total_examples=model_dm.corpus_count,epochs=1)
    model_dbow.train(sentences_perm(b), total_examples=model_dbow.corpus_count,epochs=1)


# 从我们的模型中获得训练过的向量
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


train_vecs_dm = getVecs(model_dm, x_train, size)
train_vecs_dbow = getVecs(model_dbow, x_train, size)
train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

# 训练测试数据集
c = []
c.extend(x_test)

for epoch in range(1):
    model_dm.train(sentences_perm(c), total_examples=model_dm.corpus_count,epochs=1)
    model_dbow.train(sentences_perm(c), total_examples=model_dbow.corpus_count,epochs=1)

# 创建测试数据集向量
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)
test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy: %.2f' % lr.score(test_vecs, y_test)