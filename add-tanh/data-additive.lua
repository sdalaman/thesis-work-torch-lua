local stringx = require('pl.stringx')
local file = require('pl.file')
require 'torch'

function table.slice(tbl, first, last, step)
  local sliced = {}
  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end
  return sliced
end


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
  self.sntclen={}
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

function Corpus:paddingLSTM(sequence)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , math.min(self.longest_seq,#sequence) do
    new_sequence[i] = sequence[i]
  end
  for i = math.min(self.longest_seq,#sequence)+1, self.longest_seq do
    new_sequence[i] = '<pad>'
  end
  return new_sequence
end


function Corpus:load_data(fname,max_len)
   local data = file.read(fname)
   -- data = stringx.replace(data, '\n', '<eos>')
   self.sequences = stringx.split(data, '\n')
   --all_words = stringx.split(data,self.sep) -- tokenized with tab
   print(string.format("Loading %s, Number of sentences = %d", fname, #self.sequences))
   self.no_of_sents = #self.sequences
   for i = 1 , #self.sequences do
     local words = stringx.split(self.sequences[i],self.sep) -- tokenized with tab
     words_len = #words
     self.sntclen[i] = words_len
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
  local totseq=0
   for i = 1, #self.sequences do
     local words = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     words = self:padding(words,longest_seq)
     totseq=totseq+#words
     for j = 1, #words do 
       if self.vocab_map[words[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[words[j]] = self.vocab_idx
       end
       x[i][j] = self.vocab_map[words[j]] 
     end
   end
   print(string.format("Number of words = %d , mean sntc len %4.2f", self.vocab_idx,totseq/#self.sequences))
   self.no_of_words = self.vocab_idx
   return x
end

function Corpus:shape_dataLSTM()
  local x = torch.zeros(#self.sequences, self.longest_seq)
  local totseq=0
   for i = 1, #self.sequences do
     local words = stringx.split(self.sequences[i], self.sep) -- tokenized with tab
     totseq=totseq+#words
     words = self:paddingLSTM(words)
     for j = 1, #words do 
       if self.vocab_map[words[j]] == nil then
         self.vocab_idx = self.vocab_idx + 1
         self.vocab_map[words[j]] = self.vocab_idx
       end
       x[i][j] = self.vocab_map[words[j]] 
     end
   end
   print(string.format("Number of words = %d , mean sntc len %4.2f", self.vocab_idx,totseq/#self.sequences))
   self.no_of_words = self.vocab_idx
   return x
end


function Corpus:prepare_corpus(max_len)
   self:load_data(self.data_file,max_len)
end

function Corpus:get_data()
   self.x = self:shape_data()
   return self.x
end

function Corpus:get_dataLSTM()
   self.x = self:shape_dataLSTM()
   return self.x
end

function Corpus:ret_data()
   return self.x
end

