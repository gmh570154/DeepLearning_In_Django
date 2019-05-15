from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec

with open('twitter_data/pos_tweets.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open('twitter_data/neg_tweets.txt', 'r') as infile:
    neg_tweets = infile.readlines()

# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)


# 零星的预处理
def cleanText(corpus):
    corpus = [z.lower().replace('n', '').split() for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)

n_dim = 300
# 初始化模型并创建词汇表（vocab）
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)

# 训练模型 (会花费几分钟)
imdb_w2v.train(x_train)

