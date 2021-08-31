# Systematic tissue annotations of genomic samples by modeling unstructured metadata
#
# Nathaniel T. Hawkins, Marc Maldaver, Anna Yannakopoulos, Lindsay A. Guare, Arjun Krishnan
# Corresponding Author: Nathaniel T. Hawkins, hawki235@msu.edu
#
# demo.sh - demo for running txt2onto on a given set of input text to make predictions using our models
# 
# Author: Nathaniel T. Hawkins
# Date: 14 August, 2020
# Updated: 31 August, 2021

if [ ! -d ../out/ ]; then
    echo "Making directory ../out/ to send outputs to"
    mkdir ../out/
fi

python txt2onto.py --file ../data/example_input.txt --out ../out/example_output.txt --predict