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

Requirements can be installed by running `pip install -r requirements.txt`. These requirements have been verified for python verison 3.7.7.

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



### Output

DESCRIBE
### Demo

For an example, run `sh demo.sh` in the `src/` directory.

```bash
cd src/
sh demo.sh
```

## Overview of Repository

Here, we list the files we have included as part of this repository.

* `bin/` - 

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
<!--
### References

-->