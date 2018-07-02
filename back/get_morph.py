import pipeline_caller
import codecs
import sys


caller = pipeline_caller.PipelineCaller()
caller.token = 'xKktd3CBONoiwvB6okbu61W5gCoOMsTh'
caller.processing_type = 'whole'
caller.tool = 'pipelineNoisy'

inputf = codecs.open('turkish.1000.tok', 'r','utf-8')
outputf = codecs.open('turkish.1000.morph','w','utf-8')

data = inputf.read()
texts = data.split('\n')


for x in range(len(texts)):
	print(texts[x])
	caller.text = texts[x]
	result = caller.call()
	print(result)
	words = result.split('\n')
	
	sent = []
	till = -1
	for i in range(len(words)):
		parts = words[i].split('\t')
		#print(parts)
		if i == till:
			parts[2] = keep
			#print(parts[2])
			till = -1
		#print(i)
		if parts[1] == '_':
			keep = parts[2]
			till = int(parts[6])-1
			#print(till)
			continue
		
		#print(parts)	 
		if parts[5] != '_':
			morphs = parts[5].split('|')
			if parts[2] == '_' or parts[1] == parts[2]:
				word = parts[1]+' '+' '.join(morphs)
			else:
				word = parts[1]+' '+parts[2]+' '+' '.join(morphs)
		else:
			if parts[2] == '_' or parts[1] == parts[2]:
				word = parts[1]+' '+parts[4]
			else:
				word = parts[1]+' '+parts[2]+' '+parts[4]
	
		sent.append(word)
	
	outputf.write('\t'.join(sent)+'\n')
	sys.stdout.write("\r%d%%" % x)
	sys.stdout.flush()
	
