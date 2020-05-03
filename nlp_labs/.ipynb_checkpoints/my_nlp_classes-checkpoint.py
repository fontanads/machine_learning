from operator import itemgetter

import nltk
import re
from string import punctuation

import wikipedia  #!pip install wikipedia

regex_punct = re.compile('[%s]' % re.escape(punctuation))
regex_alphabet = re.compile('/w+/u')


class Wikipedia_Crawler():

    def __init__(self,query_list=None, max_results_per_query=1, max_len_vocabulary=1000, skip_top_words=0,max_len_sentence=200):
        self.query_list            = query_list
        self.max_results_per_query = max_results_per_query
        self.max_len_vocabulary    = max_len_vocabulary
        self.skip_top_words        = skip_top_words
        self.max_len_sentence      = max_len_sentence
        self.tag_start = '<START>'
        self.tag_oov   = '<OOV>'
        self.tag_end   = '<END>'
        self.buffer()
        
        
    def buffer(self):
        self.titles_list     = []
        self.unique_tokens   = set()
        self.word_count      = {}
        self.sentences       = []
        self.coded_sentences = []
        
    # RUN 1
    def query_wiki(self):
        for query in self.query_list:
            self.titles_list.extend(wikipedia.search(query, results=self.max_results_per_query, suggestion=False))

    # RUN 2
    def get_wiki_sentences(self):
        for title in self.titles_list:
            wiki = wikipedia.page(title=title, pageid=None, auto_suggest=True, redirect=True, preload=False)
            new_sentences = self.my_sentence_tokenizer(wiki.content)
            self.sentences.extend(new_sentences) # if argument is list, each item goes to sentences, not list as an item
    
    # RUN 3
    def get_all_tokens(self):
        for s in self.sentences:
            tokens = self.my_tokenizer(s)
            self.unique_tokens.update(set(tokens))
    
    # RUN 4
    def count_vocabulary(self):
        for s in self.sentences:
            tokens = self.my_tokenizer(s)
            for t in tokens:
                self.word_count[t] = 1 if t not in self.word_count else self.word_count[t] + 1
        # sort words by absolute frequency in tuples
        self.word_frequency = sorted(self.word_count.items(), key=itemgetter(1), reverse=True)  
    
    # RUN 5:
    def generate_vocabulary(self,max_len_vocabulary=None,skip_top_words=None):
        if max_len_vocabulary is not None:
            self.max_len_vocabulary = max_len_vocabulary
            
        if skip_top_words is not None:
            self.skip_top_words = skip_top_words
            
        self.word2idx      = {self.tag_oov:0, self.tag_start:1, self.tag_end:2}
        self.idx2word      = {0:self.tag_oov, 1:self.tag_start, 2:self.tag_end}
        
        V = min(self.max_len_vocabulary, len(self.unique_tokens)-self.skip_top_words) # resize V if vocabulary is too small
        
        for w, i in self.word_frequency[self.skip_top_words:self.skip_top_words+V]:
            self.idx2word[len(self.word2idx)] = w 
            self.word2idx[w] = len(self.word2idx)
        
    # RUN 6:
    def encode_all_sentence(self,max_len_sentence=None,pad_sentences=True):
        self.coded_sentences = []

        if max_len_sentence is not None:
            self.max_len_sentence = max_len_sentence
            
        for s in self.sentences:
            coded_s = self.encode_one_sentence(s,pad_sentences)
            self.coded_sentences.append(coded_s)
    
    # helper methods, maybe belong to a different class
    
    def decode_one_sentence(self,c):
        decoded_s = [self.idx2word[t] for t in c if t>=1] # t>2 does not include <START> or <END>; t>=1 does not include <OOV>
        return decoded_s
    
    def encode_one_sentence(self,s,pad_sentences):
        coded_s = [self.word2idx[self.tag_start]]
        for t in self.my_tokenizer(s):
            coded_t = self.word2idx[t] if t in self.word2idx else self.word2idx[self.tag_oov]
            coded_s.append(coded_t)
            if len(coded_s)==self.max_len_sentence+1:
                break
                
        coded_s.append(self.word2idx[self.tag_end])    
        if pad_sentences:
            coded_s = self.pad_one_sentence(coded_s)
            
        return coded_s
    
    def pad_one_sentence(self,coded_s):
        coded_s += [self.word2idx[self.tag_oov]] * (self.max_len_sentence+2 -len(coded_s))
        return coded_s
    
    def my_sentence_tokenizer(self,raw_text):
        doc = raw_text.lower()
        doc = nltk.tokenize.sent_tokenize(doc)
        return doc
    
    def my_tokenizer(self,s):
        s = s.lower()
        s = regex_punct.sub(" ",s)
        s = regex_alphabet.sub(" ",s)
        tokens = nltk.tokenize.wordpunct_tokenize(s)
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)]  # remove tokens com dígitos numéricos
        return tokens