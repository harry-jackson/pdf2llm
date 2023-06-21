import boolean
import re
from shlex import split
import numpy as np
import pickle

def regex_flags(case_sensitive = False):
    flags = re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE
    return flags

def make_regex_search_function(query, case_sensitive = False):
    flags = regex_flags(case_sensitive)

    regex = re.compile(query, flags = flags)

    def search_function(text):
        if regex.search(text):
            return True
        return False
    
    return search_function

def make_simple_search_function(query, case_sensitive = False):
    # Add word boundaries, replace spaces with optional whitespace patterns
    regex_pattern = r'\b' + query.replace(' ', r'\s+').replace('*', '\w*?') + r'\b'

    return make_regex_search_function(regex_pattern, case_sensitive = case_sensitive)

def handle_wildcards(pattern):
    pattern = re.sub(' ', '\\\s+', pattern)
    pattern = pattern.replace("*", "\w*")  # '*' Matches any characters
    pattern = pattern.replace("?", "\w")   # '?' Matches a single character
    return r'\b{}\b'.format(pattern)        # \b ensures we are matching whole words

def evaluate(expr, symbol_values):
    if isinstance(expr, boolean.Symbol):
        return symbol_values[expr.obj]
    elif isinstance(expr, boolean.NOT):
        return not evaluate(expr.args[0], symbol_values)
    elif isinstance(expr, boolean.AND):
        return all(evaluate(arg, symbol_values) for arg in expr.args)
    elif isinstance(expr, boolean.OR):
        return any(evaluate(arg, symbol_values) for arg in expr.args)
    else:
        return expr.obj

def tokenize_expression(expr, case_sensitive = False):
    special_tokens = ('(', ')', 'AND', 'OR', 'NOT')

    expr = expr.replace('(', ' ( ').replace(')', ' ) ')
    tokens = split(expr)

    for i in range(len(tokens) - 1, 0, -1):
        if not tokens[i].upper() in special_tokens and not tokens[i-1].upper() in special_tokens:
            tokens.insert(i, 'AND')

    new_expr = ''
    token_map = {}
    i = 0
    for token in tokens:
        if token.upper() in ('(', ')', 'AND', 'OR', 'NOT'):
            new_expr += token + ' '
        else:
            token_alias = 'x_' + str(i)
            new_expr += token_alias + ' '

            flags = regex_flags(case_sensitive)

            token_map[token_alias] = re.compile(handle_wildcards(token), flags = flags)
            i += 1
    return (new_expr[:-1], token_map)

def make_boolean_search_function(expr, case_sensitive = False):
    
    if not expr:
        return lambda text: True

    translated_expr, token_map = tokenize_expression(expr, case_sensitive = case_sensitive)

    # Parse boolean expression
    bool_alg = boolean.BooleanAlgebra()
    bool_expr = bool_alg.parse(translated_expr)

    def search_function(text):
        # Dictionary to store boolean values of each unique symbol in the expression
        symbol_values = {}
        for symbol in bool_expr.symbols:
            pattern = token_map[str(symbol)]
            
            symbol_values[symbol.obj] = bool(pattern.search(text))

        # Evaluate boolean expression with symbol values
        return evaluate(bool_expr, symbol_values)
    
    return search_function

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