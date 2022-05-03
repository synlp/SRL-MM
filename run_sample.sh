
# train
python srl_main.py --do_train --train_data_path=./data/demo/train.stanford.json  --dev_data_path=./data/demo/dev.stanford.json --test_data_path=./data/demo/test.stanford.json --use_xlnet --bert_model ./XLNet_base_cased  --use_crf --n_mlp=200 --max_seq_length=300 --train_batch_size=2 --direct --eval_batch_size=2 --num_train_epochs=2 --warmup_proportion=0.2 --learning_rate=1e-5 --patient=100 --model_name=sample_model_kvmn_dep --knowledge=dep


# test
python srl_main.py --do_test --eval_model=./models/test_model/model --test_data_path=./data/demo/test.stanford.json --eval_batch_size=2

