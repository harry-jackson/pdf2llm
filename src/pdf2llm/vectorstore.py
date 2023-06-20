import numpy as np
import pickle

class VectorStore:
    
    def __init__(self, embedding_function = None, vectors = None, labels = np.array([], dtype = str)):
        self.EmbeddingFunction = embedding_function
        self.Vectors = vectors
        self.Labels = labels
        
    def add(self, texts, ids):
        
        self.Labels = np.concatenate([self.Labels, ids])
        vectors = [self.EmbeddingFunction(t) for t in texts]
        matrix = np.vstack(vectors)
        if self.Vectors == None:
            self.Vectors = matrix
        else:
            self.Vectors = np.concatenate([self.Vectors, matrix], axis = 0)
        
    def filter_vectors(self, ids):
        ids = np.array(ids)
        filter_ids = np.isin(self.Labels, ids)
        filtered_labels = self.Labels[filter_ids]
        filtered_vectors = self.Vectors[filter_ids, ]
        return VectorStore(filtered_vectors, filtered_labels)
        
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def search(self, text, n = 5):
        v = self.EmbeddingFunction(text)
        similarities = np.dot(self.Vectors, v)
        res = sorted(zip(self.Labels, similarities), key = lambda x: x[1], reverse = True)[:n]
        return [r[0] for r in res]
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            store = pickle.load(f)
            
        return store