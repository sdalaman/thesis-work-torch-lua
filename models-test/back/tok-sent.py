import nltk
import codecs
from nltk.tokenize import sent_tokenize

inpf1 = codecs.open('71', 'r','utf-8')
outf1 = codecs.open('71.sent','w','utf-8')
inpf2 = codecs.open('32', 'r','utf-8')
outf2 = codecs.open('32.sent','w','utf-8')


data1 = inpf1.read()
data2 = inpf2.read()

sent_tokenize_list1 = sent_tokenize(data1)
sent_tokenize_list2 = sent_tokenize(data2)
l1 = len(sent_tokenize_list1)
l2 = len(sent_tokenize_list2)
print(" en %d tr %d" % (l1,l2))

for i in range(l1):
	outf1.write(sent_tokenize_list1[i]+'\n')

for i in range(l2):
	outf2.write(sent_tokenize_list2[i]+'\n')


