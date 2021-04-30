# Systematic tissue annotations of –omics samples by modeling unstructured metadata

Nathaniel T. Hawkins, Marc Maldaver, Lindsay A. Guare, Arjun Krishnan

There are currently >1.3 million human –omics samples that are publicly available. However, this valuable resource remains acutely underused because discovering samples, say from a particular tissue of interest, from this ever-growing data collection is still a significant challenge. The major impediment is that sample attributes such as tissue/cell type of origin are routinely described using non-standard, varied terminologies written in unstructured natural language. Here, we provide a natural-language-processing-based machine learning approach (NLP-ML) to infer tissue and cell type annotations for –omics samples based only on their free-text metadata. NLP-ML works by creating numerical representations of sample text descriptions and using these representations as features in a supervised learning classifier that predicts tissue/cell-type terms in a structured ontology. Our approach significantly outperforms representative methods of existing state of the art approaches to addressing the sample annotation problem. We have also demonstrated the biological interpretability of tissue NLP-ML models using an analysis of their similarity to each other and an evaluation of their ability to classify tissue- and disease-associated biological processes based on their text descriptions alone. 

Previous studies have shown that the molecular profiles associated with –omics samples are highly predictive of a variety of sample attributes. Using transcriptome data, we have shown that NLP-ML models can be nearly as accurate as expression-based models in predicting sample tissue annotations. However, the latter (models based on –omics profiles) need to be trained anew for each –omics experiment type. On the other hand, once trained using any text-based gold-standard, approaches such as NLP-ML can be used to classify sample descriptions irrespective of sample type. We demonstrated this versatility by using NLP-ML models trained on microarray sample descriptions to classify RNA-seq, ChIP-seq, and methylation samples without retraining.

Here, we provide the fully trained models and a simple utility script for users to leverage the predictive power of NLP-ML to annotate their text corpora of interest for 346 tissues and cell types from [UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon).  These NLP-ML models are trained using our full gold standard. As a note: in our manuscript, we discussed the results of models who had sufficient training data in the gold standard for at least 3-fold CV. The remaining models were not discussed or examined in detail in our work due to lack of sufficient labeled samples. We have included trained models for all available tissues and cell types so as to provide users with the maximum amount of predictive capability. However, it should be noted that some models included in this repository have very little training data (i.e., small number of positively labeled examples) and thus may provide inaccurate annotations. The full list of cross-validated models can be found [here](https://github.com/krishnanlab/NLP-ML_Annotation/blob/main/gold_standard/CrossValidatedModels.txt), and the full list of models presented in our paper can be found [here](https://github.com/krishnanlab/NLP-ML_Annotation/blob/main/gold_standard/ManuscriptModels.txt).

<!--
### Link to Paper

A link to our paper can be found [HERE]().
-->

## Installation

Requirements can be installed by running `pip install -r requirements.txt`. These requirements have been verified for python version 3.7.7.

If you are using a newer version of python or scikit-learn than listed in `requirements.txt`, you may see a warning message like the following: 

```
UserWarning: Trying to unpickle estimator LogisticRegression from version 0.23.1 
when using version <YOUR CURRENT VERSION>. This might lead to breaking code or invalid results. 
Use at your own risk. warnings.warn()
```

We recommend using the stated versions of the libraries in `requirements.txt` to avoid any potential issues, however, in our testing with newer versions of scikit-learn, we have encountered no problems. 

## Usage

### Input

The input should be a plain text file with one description per line. An example is provided [here](https://github.com/krishnanlab/NLP-ML_Annotation/blob/main/data/example_input.txt) with a small excerpt below.

```
na colon homo sapiens colonoscopy male adenocarcinoma extract total rna le biotin norm specified colonoscopy male adenocarcinoma specified ...
homo sapiens adult allele multiple sclerosis clinically isolated syndrome none peripheral blood mononuclear human pbmc non treated sampling ...
wholeorganism homo sapiens prostate prostate patient id sample type ttumor biopsy ctrlautopsy sample percentage tumor prostate patient ...
normal adult patient normal adult patient age gender male skeletal muscle homo sapiens unknown extract total rna le biotin norm unknown ...
medium lp stimulation blood lp homo sapiens myeloid monocytic cell medium lp stimulation extract total rna le biotin norm medium lp stimulation ...
```

The input text will be preprocessed during the execution of `src/main.py`. For more information on the preprocessing pipeline, see the `preprocess` function in `src/utils.py`.

### Output

The prediction task can then be performed by running:

```
python main.py --file /path/to/text/file.txt --out /path/to/write/embeddings/to.txt --predict
```

This will read in the input text from `path/to/text/file.txt`, create a word embedding for each line of text and write it to `/path/to/write/embeddings/to.txt`, and then make a prediction for each line of text for each of our models and write it to `/path/to/write/embeddings/predictions_to.txt`. The output path for predicted probabilities is automatically generated when the flag is passed. The i,j entry of the output dataframe is the predicted probability assigned by model i for text snippet j from the input file. 

Alternatively, a single text snippet can be read from the command line:

```
python main.py --text SOME SAMPLE DESCRIPTION OR PIECE OF TEXT --out /path/to/write/embeddings/to.txt --predict
```

Which will write a single word embedding to `/path/to/write/embeddings/to.txt` and write the predictions to `/path/to/write/embeddings/predictions_to.txt`. 

If the user only wants word embeddings, the `--predict` flag can be omitted. Word embeddings are always generated and written to file whether predictions are made or not.

### Demo

For an example, run `sh demo.sh` in the `src/` directory.

```bash
cd src/
sh demo.sh
```

This will read in the example input file from `data/example_input.txt`, write embeddings to `out/example_output.txt`, and write predictions to `out/predictions_example_output.txt`.

## Overview of Repository

Here, we list the files we have included as part of this repository.

* `bin/` - The fully trained Logistic Regression models stored as pickle (`.p`) files
* `data/` - Example input file and files needed for making embeddings and output predictions
    * `data/UBERONCL.txt` - A text file that maps the model ontology identifiers to plain text
    * `data/pubmed_weights.txt.gz` - IDF weights for every unique word across PubMed used to make a weighted average embedding for each piece of text
* `gold_standard/` - Raw datafiles from our manuscript
    * `gold_standard/AnatomicalSystemsPerModel.json` - Mapping of every term in UBERON to a high-level anatomical system
    * `gold_standard/CrossValidatedModels.txt` - A list of models we had sufficient positively labeled training data to perform cross validation on
    * `gold_standard/GoldStandardLabelMatrix_PlainText.csv` - Our manually annotated gold standard in plain text
    * `gold_standard/GoldStandardLabelMatrix.csv` - Our manually annotated gold standard with ontology identifiers
    * `gold_standard/GoldStandard_Propagated.txt` - Our manually annotated gold standard with a list of annotations for each sample not in matrix form
    * `gold_standard/GoldStandard_Sample-Descriptions.txt` - Sample descriptions for the samples in our gold standard
    * `gold_standard/GoldStandard_Sample-IDS.txt` - Sample and experiment labels corresponding to `gold_standard/GoldStandard_Sample-Descriptions.txt`
    * `gold_standard/GoldStandard_Unpropagated.txt` - The original gold standard manual annotations [1]
    * `gold_standard/ManuscriptModels.txt` - A list of the models we evaluated and showed results for in our manuscript, a subset of `gold_standard/CrossValidatedModels.txt`
    * `gold_standard/ModelsPerAnatomicalSystem.json` - Mapping that lists the tissues and cell types that belong to each high-level anatomical system
* `src/` - Main source directory
    * `src/demo.sh` - Runs an example of the pipeline
    * `src/main.py` - Primary file for making predictions on input text
    * `src/utils.py` - Utility file containing tools for making predictions on input text
* `out/` - Example directory to send outputs to

## Additional Information

### Support
For support please contact [Nat Hawkins](hawki235@msu.edu).

### Inquiry
All general inquiries should be directed to [Dr. Arjun Krishnan](arjun@msu.edu).

### License
This repository and all its contents are released under the [Creative Commons License: Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode); See [LICENSE.md](https://github.com/krishnanlab/NLP-ML_Annotation/blob/main/LICENSE).


### Funding

This work was primarily supported by US National Institutes of Health (NIH) grants R35 GM128765 to AK and in part by MSU start-up funds to AK and MSU Rasmussen Doctoral Recruitment Award and Engineering Distinguished Fellowship to NTH.

### Acknowledgements

The authors would like to thank [Kayla Johnson](https://sites.google.com/view/kaylajohnson/home) and [Anna Yannakopoulos](https://yannakopoulos.github.io/) for their feedback and contributions to this manuscript as well as their support on this research.

<!--
### Citation

-->
### References

[1] : **Ontology-aware classification of tissue and cell-type signals in gene expression profiles across platforms and technologies**. Lee Y, Krishnan A, Zhu Q, Troyanskaya OG. Bioinformatics (2013) 29:3036-3044.