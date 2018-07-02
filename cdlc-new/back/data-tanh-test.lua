local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'
require 'math'

local Corpus = torch.class("Corpus")

function Corpus:__init(fname,d_path)
  self.vocab_idx = 0
  self.vocab_map = {['<pad>'] = 0}
  self.longest_seq = 0
  self.data_file = d_path..fname
  self.x = nil
  self.no_of_words = 0
  self.no_of_sents = 0
  self. sep = '\t'
end

function Corpus:padding(sequence)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , self.longest_seq - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (self.longest_seq - math.min(self.longest_seq,#sequence))+1, self.longest_seq do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end


function Corpus:load_data(fname,max_len,max_sent)
   math.randomseed(os.time())
   local data = file.read(fname)
   -- data = stringx.replace(data, '\n', '<eos>')
   tmpsequences = stringx.split(data, '\n')
   if max_sent > #tmpsequences then
     max_sent = #tmpsequences
     print("max sent is updated to "..max_sent.."\n")
   end
   self.sequences = {}
   rlist = false
   if #sellist == 0 then
     rlist = true
   end
   for i=1,max_sent do
      if rlist then
        ln = math.random(1,#tmpsequences)
        sellist[i] = ln
      end
      self.sequences[#self.sequences + 1] = tmpsequences[sellist[i]]
   end
   --print("selected lines..\n")
   --print(sellist)
   --all_words = stringx.split(data,self.sep) -- tokenized with tab
   print(string.format("Loading %s, Number of sentences = %d", fname, #self.sequences))
   self.no_of_sents = #self.sequences
   for i = 1 , #self.sequences do
     local words = stringx.split(self.sequences[i],self.sep) -- tokenized with tab
     words_len = #words
     if self.longest_seq < words_len then 
       self.longest_seq = words_len
     end
   end
   print(string.format("Longest sequence = %d", self.longest_seq))
   if max_len ~= nil then
     self.longest_seq = math.min(self.longest_seq,max_len)
     print(string.format("Reduced down to = %d", self.longest_seq))
   end
   
end

function Corpus:shape_data()
  local x = torch.zeros(#self.sequences, self.longest_seq)
   for i = 1, #self.sequences do
     local words = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     words = self:padding(words)
     for j = 1, #words do 
       if self.vocab_map[words[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[words[j]] = self.vocab_idx
       end
       x[i][j] = self.vocab_map[words[j]] 
     end

   end
   print(string.format("Number of words = %d", self.vocab_idx))
   self.no_of_words = self.vocab_idx
   return x
end


function Corpus:prepare_corpus(max_len,max_sent)
   self:load_data(self.data_file,max_len,max_sent)
end

function Corpus:get_data()
   self.x = self:shape_data()
   return self.x
end


function testcorpus_shapedata(corpus,test_corpus)
  local x = torch.zeros(#test_corpus.sequences, test_corpus.longest_seq)
   for i = 1, #test_corpus.sequences do
     local words = stringx.split(test_corpus.sequences[i], test_corpus.sep) -- tokenized with tab
     words = test_corpus:padding(words)
     for j = 1, #words do 
       if corpus.vocab_map[words[j]] == nil then
         x[i][j] = 0 
      else
         x[i][j] = corpus.vocab_map[words[j]] 
      end
     end

   end
   return x
end
