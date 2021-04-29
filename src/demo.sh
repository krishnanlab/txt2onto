if [ ! -d ../out/ ]; then
    echo "Making directory ../out/ to send outputs to"
    mkdir ../out/
fi

python main.py --file ../data/example_input.txt --out ../out/example_output.txt --predict