# SRL-MM

This is the implementation of [Syntax-driven Approach for Semantic Role Labeling]() at LREC2022.

Please contact us at `yhtian@uw.edu` if you have any questions.

## Citation

If you use or extend our work, please cite our paper at LREC2022.

```

```

## Requirements

Our code works with the following environment.
* `python=3.7`
* `pytorch=1.7`

## Downloading BERT and XLNet

In our paper, we use [BERT](https://github.com/google-research/bert) and [XLNet](https://github.com/zihangdai/xlnet) as the encoder.
We follow the [instructions](https://huggingface.co/docs/transformers/converting_tensorflow_models) to convert the TensorFlow checkpoints to the PyTorch version.

**Note**: for XLNet, it is possible that the resulting `config.json` misses the hyper-parameter `n_token`. You can manually add it and set its value to `32000` (which is identical to `vocab_size`).

## Datasets

We use [CoNLL 2005](https://www.cs.upc.edu/~srlconll/) and [CoNLL 2012](https://conll.cemantix.org/2012/data.html) in our paper.

To obtain and pre-process the data, please go to `data_processing` directory for more information.

All processed data will appear in `data` directory.

## Train and Test the model

You can find the command lines to train and test models on a small sample data in `run_sample.sh`.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--use_xlnet`: use XLNet as encoder.
* `--bert_model`: the directory of pre-trained BERT/XLNet model.
* `--knowledge`: the knowledge type to be used. It should be one of `pos`, `syn`, and `dep`.
* `--use_crf`: use CRF after the bi-affine attentions.
* `--model_name`: the name of model to save.
