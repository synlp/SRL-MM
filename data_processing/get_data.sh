mkdir ./tmp/

# get conll 05

./scripts/fetch_and_make_conll05_data.sh /path/to/LDC99T42/RAW

mkdir ./tmp/cn05
mkdir ./tmp/cn05/json/

cp data/srl/train.english.conll05.jsonlines ./tmp/cn05/json/train.json
cp data/srl/dev.english.conll05.jsonlines ./tmp/cn05/json/dev.json
cp data/srl/test_wsj.english.conll05.jsonlines ./tmp/cn05/json/test.json
cp data/srl/test_brown.english.conll05.jsonlines ./tmp/cn05/json/brown.json

mkdir ./tmp/cn05/tsv/
python process_SRL.py ./tmp/cn05/

mkdir ../data/CN05

cp ./tmp/cn05/tsv/*.tsv ../data/CN05


# get conll 12
./scripts/make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0

mkdir ./tmp/cn12
mkdir ./tmp/cn12/json/

cp data/srl/train.english.mtl.jsonlines ./tmp/cn12/json/train.json
cp data/srl/dev.english.mtl.jsonlines ./tmp/cn12/json/dev.json
cp data/srl/test.english.mtl.jsonlines ./tmp/cn12/json/test.json

mkdir ./tmp/cn12/tsv/
python process_SRL.py ./tmp/cn12/

mkdir ../data/CN12

cp ./tmp/cn12/tsv/*.tsv ../data/CN12
