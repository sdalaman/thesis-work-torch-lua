import nltk
import codecs

inpf1 = codecs.open('turkish.all', 'r','utf-8')
outf1 = codecs.open('turkish.all.tok','w','utf-8')
inpf2 = codecs.open('english.all', 'r','utf-8')
outf2 = codecs.open('english.all.tok','w','utf-8')
max_len = 100

data1 = inpf1.read()
sents1 = data1.split('\n')

data2 = inpf2.read()
sents2 = data2.split('\n')

no_of_sent = len(sents1)
for i in range(no_of_sent):
	tokens1 = nltk.word_tokenize(sents1[i])
	tokens2 = nltk.word_tokenize(sents2[i])
	if len(tokens1) < max_len and len(tokens2) < max_len:
		outf1.write('\t'.join(tokens1)+'\n')
		outf2.write('\t'.join(tokens2)+'\n')



