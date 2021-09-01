# Systematic tissue annotations of genomic samples by modeling unstructured metadata
#
# Nathaniel T. Hawkins, Marc Maldaver, Anna Yannakopoulos, Lindsay A. Guare, Arjun Krishnan
# Corresponding Author: Nathaniel T. Hawkins, hawki235@msu.edu
#
# input.py - script to create an example input from our gold standard for training 
# an NLP-ML model
# 
# Author: Nathaniel T. Hawkins
# Date: 31 August, 2021

## Imports
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    ## Command line arguments
    parser = ArgumentParser()
    parser.add_argument("--ont", 
                        help = "Ontology term to build a model for", 
                        required = True)
    parser.add_argument("--out",
                        help = "Directory to save trained model to")
    args = parser.parse_args()


    ## Make output directory if it does not exist
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    ## Load in gold standard labels
    gold_std_labels = pd.read_csv("../gold_standard/GoldStandard_LabelMatrix.csv", index_col = 0)
    gold_std_labels.set_index("Sample_ID", inplace = True)
    gold_std_labels.drop("Experiment_ID", inplace = True, axis = 1)

    ## Check for ontology terms in gold std
    if args.ont not in gold_std_labels.columns:
        raise ValueError(f"No true labels for {args.ont}")

    ## Load in text
    with open("../gold_standard/GoldStandard_Sample-Descriptions.txt", "r") as f:
        text_descriptions = f.readlines()
    with open("../gold_standard/GoldStandard_Sample-IDS.txt", "r") as f:
        text_ids = f.readlines()
    gold_std_text = {text_ids[i].strip().split("\t")[0]:text_descriptions[i].strip() for i in range(len(text_ids))}

    ## Create training input
    ## Get labels for model from gold standard
    labels_df   = gold_std_labels[args.ont]
    labels_gsm  = labels_df.index
    labels_labs = labels_df.values

    ## Filter out the ignore terms
    labels_gsm  = labels_gsm[labels_labs != 0]
    labels_labs = labels_labs[labels_labs != 0]

    ## Construct output to be used to train machine model
    output = []
    for gsm, lab in zip(labels_gsm, labels_labs):
        output.append(f"{gsm}\t{int(lab)}\t{gold_std_text[gsm]}\n")

    ## Save output to file
    with open(f"{args.out}{args.ont.replace(':', '-')}_NLP-ML-input.txt", "w") as f:
        for line in output:
            f.write(line)