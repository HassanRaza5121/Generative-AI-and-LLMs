from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize,sent_tokenize
import torch
import numpy as np
import torch.nn as nn
class Embeddings:
    def __init__(self,file_name) -> None:
        self.file_name = file_name
    def get_data(self):
        with open(self.file_name,'rb') as file:
            corpus = file.readlines()
        return corpus
    def tokenization(self):
        corpus = self.get_data()
        tokens = word_tokenize(corpus)
        return tokens
    def word_embeddings(self):
        tokens = self.tokenization()
        model = Word2Vec(sentences=tokens,vector_size=100,min_count=1,window=5)
        model.save("word2vec.model")
file_name = "c:\Users\ST\Desktop\Why FastText Outperforms Word2Vec.docx"
ggggobj gggggggggggggggggggggggggg= Embeddings       
10'hg
gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg\