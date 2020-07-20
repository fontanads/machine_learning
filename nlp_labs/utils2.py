import nltk
import itertools

def my_word_tokenizer(s,tag_start='<s>',tag_end='</s>',tag=True,stopwords=[]):
    s_tokens = nltk.word_tokenize(s) # initial tokenization
    
    if tag:
        s_tokens = [tag_start] + [t.lower() for t in s_tokens if (t.isalpha() or t == '.') and  (t not in stopwords)] +  [tag_end]
    else:
        s_tokens = [t.lower() for t in s_tokens if (t.isalpha() or t == '.') and (t not in stopwords)]
    
    return s_tokens

def my_sentence_tokenizer(raw_text,tag_start='<s>',tag_end='</s>',tag=True, stopwords=[]):
    doc = raw_text.lower()
    doc = nltk.tokenize.sent_tokenize(doc)
    new_doc = []
    for s in doc:
        new_doc.append(my_word_tokenizer(s,tag_start,tag_end,tag=tag,stopwords=stopwords))
    return new_doc

def get_all_tokens(doc):
    return list(itertools.chain.from_iterable(doc))

def pad_one_coded_sentence(coded_s, word2idx, max_len_sentence=10, tag_pad='<pad>'):
    coded_s += [word2idx[tag_pad]] * (max_len_sentence +2 - len(coded_s))
    return coded_s

def encode_sentence(s, word2idx,max_len_sentence=10, pad_sentences=True,
                    tag_oov='<unk>', tag_pad='<pad>'):
    coded_s = []
    for t in s:
        coded_t = word2idx[t] if t in word2idx else word2idx[tag_oov]
        coded_s.append(coded_t)
        if len(coded_s)==max_len_sentence+1:
            break
    
    if pad_sentences:
        coded_s = pad_one_coded_sentence(coded_s,word2idx,max_len_sentence,tag_pad)
            
    return coded_s