# Get Stanford CoreNLP

# This only shows an example, you can use other versions of Stanford CoreNLP.

wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
upzip http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip

# process CoNLL 05 and CoNLL 12

./get_syntax.sh --dataset=./data/CN05
./get_syntax.sh --dataset=./data/CN12

