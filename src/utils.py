# Systematic tissue annotations of genomic samples by modeling unstructured metadata
#
# Nathaniel T. Hawkins, Marc Maldaver, Lindsay A. Guare, Arjun Krishnan
# Corresponding Author: Nathaniel T. Hawkins, hawki235@msu.edu
#
# utils.py - script containing functions utilized in main.py
# 
# Author: Nathaniel T. Hawkins
# With significant contributions by: Anna Yannakopoulos
# Date: 14 August, 2020

## Imports 
import pickle
import os
import gzip
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, BertEmbeddings, ELMoEmbeddings

import nltk, torch, flair

nltk.download('stopwords')
nltk.download('wordnet')
flair.device =  torch.device("cpu")




def loadDictionary(fname, compression = True, try_except = True, key_dtype = str,
                   value_dtype = float, delimiter = "\t", verbose = False):
    """
    Loads dictionary from two column text file.

    Args:
        fname (str): Path to file to load
        compression (bool, optional): Flag to denote whether the file is GZIP compressed. Defaults to True.
        try_except (bool, optional): Flag to denote try/except block for defining key value pairs. May alleviate issues with loading incorrectly formatted lines. Defaults to True.
        key_dtype ([type], optional): Data type of keys (first column). Defaults to str.
        value_dtype ([type], optional): Data type of the value (second column). Defaults to float.
        delimiter (str, optional): Character to distinguish between two columns. Defaults to "\t".
        verbose (bool, optional): Prints updates verbosely. Defaults to False.

    Raises:
        FileExistsError: Dictionary file cannot be found
        ValueError: Unable to load dictionary with specified parameters

    Returns:
        dict: Dictionary with key-value pairs read from file
    """
    if not os.path.exists(fname): 
        raise FileExistsError(f"Filename {fname} does not exist.")

    try:
        if compression:
            with gzip.open(fname, "rt") as f:
                lines = f.readlines()
        else:
            with open(fname, "r") as f:
                lines = f.readlines()
    except:
        raise ValueError(f"Unable to load dictionary from file {fname}.")
    
    return_dict_ = {}
    for i,line in enumerate(lines):
        if try_except:
            try:
                terms = line.strip().split(delimiter)
                key = key_dtype(terms[0])
                value = value_dtype(terms[1])
            except:
                if verbose: print(f"Unable to load entry [{i}]: '{line.strip()}'.")
                else: pass
        else:
            terms = line.strip().split(delimiter)
            key = key_dtype(terms[0])
            value = value_dtype(terms[1])
        return_dict_[key] = value
    return return_dict_

def loadModelFromPickle(model_fname):
    """
    Read in binary storage of model and convert into model object for
    making predictions and extracting model parameters.

    Args:
        model_fname (str): Path to pickle file of model

    Returns:
        Logisitc Regression classifier model
    """
    model_object_ = pickle.load(open(model_fname, "rb"))
    return model_object_

def makeStackedEmbedding():
    """
    Creates stacked embedding used in project. Warning: will download models which
    will take some storage space.

    Returns:
        StackedEmbedding object consisting of BERT large-uncased and ELMo trained on PubMed corpus
    """
    return StackedEmbeddings(embeddings = [BertEmbeddings("bert-large-uncased"), ELMoEmbeddings("pubmed")])

def removeUnencodedText(text):
    """
    Remove non UTF-8 characters

    Author: Anna Yannakopoulos (2020)

    Args:
        text (str): Input text

    Returns:
        str: Input text where non UTF-8 character removed
    """
    return "".join([i if ord(i) < 128 else "" for i in text])

def containsNumbers(text):
    """
    Return whether a string contains any numeric characters

    Author: Anna Yannakopoulos (2020)

    Args:
        text (str): Input text

    Returns:
        bool: Boolean which is true if any characters in the string are numeric
    """
    return bool(re.search(r"\d", text))

def isAllowedWord(word, stopwords, remove_numbers, min_word_len):
    """
    Boolean function that checks to see whether a word is allowed based
    on preprocessing rules

    Author: Anna Yannakopoulos (2020)

    Args:
        word (str): Input text (SINGLE WORD)
        stopwords (list): List of stopwords to check against
        remove_numbers (bool): Boolean value to determine how to use output of containsNumber. If False, words containing
            numeric characters will be removed (function returns False)
        min_word_len (int): Minimum length a word can be

    Returns:
        bool: Denotes whether a word satisfies all required rules
    """
    stopwords_allowed = word not in stopwords
    numbers_allowed = not (remove_numbers and containsNumbers(word))
    length_allowed = len(word) >= min_word_len
    return stopwords_allowed and numbers_allowed and length_allowed

def preprocess(text, stopwords=set(stopwords.words("english")),
               stem=False, lemmatize=False, keep_alt_forms=False,
               remove_numbers=False, min_word_len=1):
    """
    Text preprocessing pipeline

    Author: Anna Yannakopoulos (2020) and Nathaniel T. Hawkins

    Args:
        text (str): Input sring of text
        stopwords (iterable, optional): List of stopwords to check against. Defaults to set(stopwords.words("english")).
        stem (bool, optional): Perform stemming on input text. Defaults to False.
        lemmatize (bool, optional): Perform lemmatization on input text. Defaults to False.
        keep_alt_forms (bool, optional): If True, stems/lemmas will be kept if they are unique compared to the original word. Defaults to False.
        remove_numbers (bool, optional): If True, words containing numeric characters will be removed. Defaults to False.
        min_word_len (int, optional): Minimum length a word must be to be kept. Defaults to 1.

    Returns:
        str: Processed version of input string
    """

    # remove non utf-8 characters
    text = removeUnencodedText(text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # convert all whitespace to spaces for splitting
    whitespace_pattern = re.compile(r"\s+")
    text = re.sub(whitespace_pattern, " ", text)

    # lowercase the input
    text = text.lower()

    # split into words
    words = text.split(" ")

    # stem and/or lemmatize words
    # filtering stopwords, numbers, and word lengths as required
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    if stem and lemmatize:
        words = [
            [word, stemmer.stem(word), lemmatizer.lemmatize(word)]
            for word in words if isAllowedWord(
                word, stopwords, remove_numbers, min_word_len)]
    elif stem:
        words = [
            [word, stemmer.stem(word)]
            for word in words if isAllowedWord(
                word, stopwords, remove_numbers, min_word_len)]
    elif lemmatize:
        words = [
            [word, lemmatizer.lemmatize(word)]
            for word in words if isAllowedWord(
                word, stopwords, remove_numbers, min_word_len)]
    else:
        words = [
            word for word in words if isAllowedWord(
                word, stopwords, remove_numbers, min_word_len)]

    if stem or lemmatize:
        if keep_alt_forms:
            # return both original and stemmed/lemmatized words
            # as long as stems/lemmas are unique
            words = [w for word in words for w in set(word)]
        else:
            # return only requested stems/lemmas
            # if both stemming and lemmatizing, return only lemmas
            words = list(zip(*words))[-1]

    return " ".join(words)

def createEmbeddingFromText(input_text, embedding_model, weight_dictionary):
    """
    Given input string, create word embedding using our approach

    Args:
        input_text (str): Raw text input. 
        embedding_model (StackedEmbedding): Method of creating embedding from flair
        weight_dictionary (dict): Dictionary listing a weight for words to use in weighted averaging

    Returns:
        numpy.array: Weighted average of embeddings for each word in input text (after preprocessing)
    """
    ## Preprocess input text and turn into a bag of words
    processed_input_text = preprocess(input_text, lemmatize = True, remove_numbers = True, min_word_len = 1)
    processed_input_words = processed_input_text.split(" ")

    ## Get weights for all available words in weight_dictionary
    processed_word_weights = {w:weight_dictionary[w] for w in processed_input_words if w in weight_dictionary.keys()}

    ## In the event that a word does not have a weight, the mean weight across all words that
    ## have weights will be used. If no words have weights, then the mean weight across all 
    ## available weights is used
    try:
        mean_weight = np.mean(list(processed_word_weights.values()))
    except:
        mean_weight = np.mean(list(weight_dictionary.values()))

    ## Assign mean weight to all words missing one
    for word in [w for w in processed_input_words if w not in processed_word_weights.keys()]:
        processed_word_weights[word] = mean_weight

    ## Make an embedding for each word and multiply by the weight
    embeddings = []
    total_weight = 0.0
    for w in processed_input_words:
        sentence_ = Sentence(w)
        embedding_model.embed(sentence_)
        embeddings.append(processed_word_weights[w]*sentence_[0].embedding.numpy().astype(float))
        total_weight += processed_word_weights[w]
    embeddings = np.array(embeddings)
    
    return np.sum(embeddings, axis = 0)/total_weight

def loadAllModels(bin_dir = "../bin/", ontid_mapping = "../data/UBERONCL.txt"):
    """
    Load all available models for making predictions

    Args:
        bin_dir (str, optional): Directory to look for models. Defaults to "../bin/".
        ontid_mapping (str, optional): String that maps id to plain text name. Defaults to "../data/UBERONCL.txt".

    Returns:
        dict: Key is the model (str) for UBERON/CL term, value is model object
    """
    ## Load ontology mapping
    ontid_to_text = loadDictionary(ontid_mapping, compression = False, key_dtype = str, value_dtype = str)

    ## Load each model
    models_ = {}
    for fname in [os.path.join(bin_dir, f) for f in os.listdir(bin_dir)]:
        model_name = fname.split("/")[-1].split("_")[0].replace("-",":")
        models_[ontid_to_text[model_name]] = loadModelFromPickle(fname)
    return models_
