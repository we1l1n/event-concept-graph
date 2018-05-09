from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer

# word2vec bin
#wv = KeyedVectors.load_word2vec_format('numberbatch-en.txt', binary=False)
#wv.save_word2vec_format('numberbatch-en.bin',binary=True)

# annoy index
#wv = KeyedVectors.load_word2vec_format('numberbatch-en.bin',binary=True)
#annoy_index = AnnoyIndexer(wv,200)
#annoy_index.save('numberbatch-en.index')

# wv = KeyedVectors.load_word2vec_format('numberbatch-en.bin', binary=True)
# annoy_index = AnnoyIndexer()
# annoy_index.load('numberbatch-en.index')
# annoy_index.model = wv

wv = KeyedVectors.load_word2vec_format('glove.6B.300d.bin', binary=True)
annoy_index = AnnoyIndexer()
annoy_index.load('glove.6B.300d.index')
annoy_index.model = wv

wv.most_similar(positive=['football','win','organization'], topn=10, indexer=annoy_index)
wv.most_similar(positive=['football','win','nationality'], topn=10, indexer=annoy_index)