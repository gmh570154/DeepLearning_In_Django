#-*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gensim
import random
import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
LabeledSentence = gensim.models.doc2vec.LabeledSentence
path = os.getcwd()
size = 300
start = datetime.datetime.now()

with open(path[:-9] + '\\aclImdb\\train\\pos_all3.txt', 'r', encoding="utf-8") as infile:
    pos_tweets = infile.readlines()

with open(path[:-9] + '\\aclImdb\\train\\neg_all3.txt', 'r', encoding="utf-8") as infile:
    neg_tweets = infile.readlines()

with open(path[:-9] + '\\aclImdb\\train\\unsup_all3.txt', 'r', encoding="utf-8") as infile:
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


print("start clean Text")
x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)
print("end clean text")

# Gensim 的 Doc2Vec 工具要求每个文档/段落包含一个与之关联的标签。我们利用 LabeledSentence 进行处理。
# 格式形如 “TRAIN_i” 或者 “TEST_i”，其中 “i” 是假的评论索引。
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

print("start labelize")
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
print("before build")
# 对所有评论创建词汇表
model_dm.build_vocab(a)
model_dbow.build_vocab(a)
print("end build vocab")
# 多次传入数据集，通过每次滑动（shuffling）来提高准确率。
all_train_reviews = np.concatenate((x_train, unsup_reviews))
b =[]
b.extend(x_train)
b.extend(unsup_reviews)


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return (shuffled)


print("before train")
for epoch in range(3):
    model_dm.train(sentences_perm(b), total_examples=model_dm.corpus_count, epochs=1)
    print("train one end")
    model_dbow.train(sentences_perm(b), total_examples=model_dbow.corpus_count, epochs=1)
print("end all train")


# 从我们的模型中获得训练过的向量
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


print("before get vecs")
train_vecs_dm = getVecs(model_dm, x_train, size)
train_vecs_dbow = getVecs(model_dbow, x_train, size)
train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
print("end get vecs")
# 训练测试数据集
c = []
c.extend(x_test)
print("before last train")
for epoch in range(3):
    model_dm.train(sentences_perm(c), total_examples=model_dm.corpus_count,epochs=1)
    print("train two end")
    model_dbow.train(sentences_perm(c), total_examples=model_dbow.corpus_count,epochs=1)
print("end last train")
# 创建测试数据集向量
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)
test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
print("end get last vecs")
print("start show")
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))
end = datetime.datetime.now()
print("运行时长：")
print(end - start)
pred_probas = lr.predict_proba(test_vecs)[:, 1]
fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()
