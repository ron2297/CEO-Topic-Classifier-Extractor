import nltk
import numpy
import math
import os
	
def tf_idf_matrix(path):
	docs = os.listdir(path)
	n = len(docs)
	tf = {}
	df = {}
	length = {}
	vocab = set()
	for doc in docs:
		file = open(path + doc, 'r')
		text = file.read()
		file.close()
		words = nltk.word_tokenize(text)
		length[doc] = 0
		for word in words:
			if not word.isalpha():
				continue
			length[doc] += 1
			word = word.lower()
			vocab.add(word)
			if (word, doc) in tf:
				tf[word, doc] += 1
			else:
				tf[word, doc] = 1
			if word in df:
				df[word].add(doc)
			else:
				df[word] = set([doc])
				
	tf_idf = {}
	for word, doc in tf:
		tf_idf[word, doc] = (tf[word, doc] / length[doc]) * math.log(n / len(df[word]), 10)
#		tf_idf[word, doc] = (1 + math.log(tf[word, doc], 10)) * math.log(n / len(df[word]), 10)
	
	return tf_idf, vocab, docs

def cos(v1, v2):
	if numpy.linalg.norm(v1) == 0 or numpy.linalg.norm(v2) == 0:
		return 0
	return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))
	

def getVector(word, tf_idf, docs):
	n = len(docs)
	v = numpy.zeros(n)
	
	for i in range(n):
		doc = docs[i]
		if (word, doc) in tf_idf:
			v[i] = tf_idf[word, doc]

	return v
	
def main():
	tf_idf, vocab, docs = tf_idf_matrix('shakespeare/')
	
	print('|V| =', len(vocab))
	
# 	v1 = getVector('fool', tf_idf, docs)
# 	v2 = getVector('wit', tf_idf, docs)
# 	print(v1)
# 	print(v2)
# 	print(cos(v1, v2))
	
	word = 'sun'
	wordVector = getVector(word, tf_idf, docs)
	
	cosines = {}
	for word2 in vocab:
		if word2 != word:
			v = getVector(word2, tf_idf, docs)
			cosines[word2] = cos(wordVector, v)
		
	c2 = [(cosines[word], word) for word in cosines]
	c2.sort()
	
	print('\nWords most similar to ' + word + ':')
	for c, w in reversed(c2[-20:]):
		print('   {0:<12} {1:<.6f}'.format(w, c))

main()
