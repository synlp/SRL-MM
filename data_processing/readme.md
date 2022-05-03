## Data Processing


### Get the data

For CoNLL 2005, you need to obtain the official [PTB3](https://catalog.ldc.upenn.edu/LDC99T42) data and the CoNLL formatted [OntoNotes 5 data](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/tree/master/conll-formatted-ontonotes-5.0).


### Process the data

`./get_data.sh` contains the script to pre-process the data. 
You may want to change the path to PTB3 and CoNLL formatted OntoNotes data accordingly.

### Obtain the syntactic features

`./get_syntax.sh` contains the script to obtain the syntactic information.
