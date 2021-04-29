# Systematic tissue annotations of genomic samples by modeling unstructured metadata
#
# Nathaniel T. Hawkins, Marc Maldaver, Lindsay A. Guare, Arjun Krishnan
# Corresponding Author: Nathaniel T. Hawkins, hawki235@msu.edu
#
# main.py - script to run new text through our models to get word embeddings
# weighted by PubMed IDF weights and predicted probabilities for all
# trained models.
# 
# Author: Nathaniel T. Hawkins
# Date: 14 August, 2020

## Imports
from utils import loadDictionary, makeStackedEmbedding, createEmbeddingFromText, loadAllModels

from argparse import ArgumentParser as AP
import numpy as np
import pandas as pd

import sys


if __name__ == "__main__":
    ## Create structure for reading file/text from the command line and writing output to file
    parser = AP()
    parser.add_argument("--text", metavar = "Input Text", nargs = "+", help = "String to create embedding for.")
    parser.add_argument("--file", metavar = "/path/to/file/in_file.txt", type = str, help = "Path to file containing text to create embeddings for. One embedding will be made for each line in input file.")
    parser.add_argument("--out", metavar = "/path/to/file/out_file.txt", type = str, help = "Path to write embeddings to file.", required = True)
    parser.add_argument("--predict", help = "Flag that denotes to predict probabilities for all available models.", action = "store_true")
    parser.add_argument("--precision", metavar = "N", default = 5, type = int, help = "Number of decimals to keep when writing to file. This applies to word embeddings and predicted probabilities.")
    args = parser.parse_args()

    ## Check to make sure that either --text or --file was passed
    ## If neither, raise ValueError
    if args.text is None and args.file is None:
        raise ValueError("Must either pass text or file containing multiple text entries.")
    
    ## Check to make sure that both --text and --file was not passed
    ## Must be one or the other
    if args.text is not None and args.file is not None:
        raise ValueError("Must pass --text _or_ --file. Cannot pass both.")
    

    ## Load IDF weights calculated by parsing PubMed in dictionary format
    print("Loading IDF weights from PubMed...")    
    weights = loadDictionary("../data/pubmed_weights.txt.gz")
    print("IDF weights from PubMed loaded!")

    ## Create stacked embedding model
    embedding_model = makeStackedEmbedding()

    ## If --predict is passed, then load all models as a dictionary
    if args.predict:
        print("Loading models to make predictions...")
        models_dict = loadAllModels()
        models = models_dict.keys()
        print(f"Successfully loaded {len(models)}!")

    ## If only a single string was passed using --text. then make word embedding and write to file
    if args.text:
        print("Creating emebedding for input text...")
        ## Create numerical representation of text
        word_vec = createEmbeddingFromText(" ".join(args.text), embedding_model = embedding_model, weight_dictionary = weights)
        print(f"Embedding successfully created! Writing to {args.out}.")

        ## Write embedding to file
        np.savetxt(args.out, np.round(word_vec, args.precision), fmt = f"%.{args.precision}e", delimiter = ",")

    ## If a file containing multiple lines of text is passed, make a word embedding for each one
    if args.file:
        ## Create list to store embeddings for lines for writing to output
        word_vectors = []

        print(f"Reading input text from {args.file}...")
        ## Read input text from file
        with open(args.file, "r") as f:
            input_text = f.readlines()
        print("Input text read in!")

        print(f"Creating word embeddings for {len(input_text)} lines of text... (This may take some time)")
        ## Create embedding for each line of text
        for line in input_text:
            word_vectors.append(createEmbeddingFromText(line, embedding_model = embedding_model, weight_dictionary = weights))
        print(f"Embeddings successfully created! Writing to {args.out}.")

        ## Turn embedding into 2D numpy array and save to file
        np.savetxt(args.out, np.round(np.array(word_vectors), args.precision), fmt = f"%.{args.precision}e", delimiter = ",")

    ## If --predict is passed, then create dataframe for predicted probabilities and run embedding(s) through
    ## models
    if args.predict:
        ## Make predictions for single word embedding
        predicted_probabilities = {}
        print("Making predictions on embedding features... (This may take some time)")

        if args.text:
            for model in models:
                predicted_probabilities[model] = models_dict[model].predict_proba(word_vec.reshape(1,-1))[-1, -1]

        ## Make predictions for multiple embeddings
        if args.file:
            for model in models:
                predicted_probabilities[model] = models_dict[model].predict_proba(word_vectors)[:, -1]
            
        print(f"Predictions made for {len(models)} models!")

        ## Turn predicted probabilities into dataframe
        pred_df = pd.DataFrame.from_dict(predicted_probabilities, orient = "index")

        ## Write predictions out to file
        terms = args.out.split("/")
        pred_out = "/".join(terms[:-1])+"/predictions_"+terms[-1]
        pred_df.round(args.precision).to_csv(pred_out)
        
