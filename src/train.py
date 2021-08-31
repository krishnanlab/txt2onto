# Systematic tissue annotations of genomic samples by modeling unstructured metadata
#
# Nathaniel T. Hawkins, Marc Maldaver, Anna Yannakopoulos, Lindsay A. Guare, Arjun Krishnan
# Corresponding Author: Nathaniel T. Hawkins, hawki235@msu.edu
#
# train.py - train an NLP-ML model for a specific tissue or cell type using
# specific file format for input (generated from input.py)
# 
# Author: Nathaniel T. Hawkins
# Date: 31 August, 2021

import os
import pickle
import numpy as np
from utils import createEmbeddingFromText, makeStackedEmbedding, loadDictionary
from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser

if __name__ == "__main__":
    ## Command line inputs
    parser = ArgumentParser()
    parser.add_argument("--input", 
                        help = "Three column, tab separated input file containing text and labels to train NLP-ML model",
                        required = True)
    parser.add_argument("--out", 
                        help = "Path to directory where trained model will be saved to",
                        required = True)
    args = parser.parse_args()

    ## Make output directory
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    ## Load in input file
    ## 3 column, TSV
    ## ID   Label   Text
    with open(args.input, "r") as f:
        input_file_contents = [x_.strip().split() for x_ in f.readlines()]
        input_ids           = [x_[0] for x_ in input_file_contents]
        input_labels        = [x_[1] for x_ in input_file_contents]
        input_text          = [x_[2] for x_ in input_file_contents]

    ## Make embedding model
    emb_ = makeStackedEmbedding()

    ## Load pubmed IDF weights
    pubmed_idf_weights = loadDictionary("../data/pubmed_weights.txt.gz")

    ## Create embedding for input text
    input_file_embeddings = []
    for i,sample_text in enumerate(input_text):
        print(f"Creating embedding: {i+1}/{len(input_text)}...")
        input_file_embeddings.append(createEmbeddingFromText(sample_text, emb_, pubmed_idf_weights))
    input_file_embeddings = np.array(input_file_embeddings)
    input_labels          = np.array(input_labels)

    ## Train logisitic regression model
    LR = LogisticRegression(penalty = "l1", C = 1, solver = 'liblinear')
    LR.fit(input_file_embeddings, input_labels)

    ## Save trained model to file
    ## Create output fname
    model_savename = args.out + "MODEL_" + args.input.split("/")[-1].split(".")[0] + ".p"
    pickle.dump(LR, open(model_savename, "wb"))