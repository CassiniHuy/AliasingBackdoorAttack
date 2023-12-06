This is the implementation of our paper "Aliasing Backdoor Attacks on Pre-trained models" (USENIX Security '23).

# Installation

## Requirements

This repo is implemented by PyTorch and you should have it installed with cuda version following the guidance from https://pytorch.org/.

You can use a docker image from NVIDIA NGC and install the required pip packages. We use the *nvcr.io/nvidia/pytorch:22.05-py3* and list the dependencies in *requirements.txt*; you can simply build it using the *Dockerfile*. 

You can install all python packages needed in *requirements.txt* and load the submodules using:

```sh
git submodule sync
git submodule update
```

or directly clone the submodule from the corresponding github repo.

## Datasets

- You can find where to download the datasets in *transfer/datasets.py*. 
- Edit the config file *transfer/datasets_configs.py* to specify where the data is placed.
- You can also edit the default fine-tuning setting in *transfer/finetune_config.py* if you need.

## Models

All the image models are from the [Timm library](https://github.com/huggingface/pytorch-image-models) except the *resnet50_miil_21k* model and facenet. 

You can find various pre-trained image models on https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv. 
To obtain the model notation to load a model, you can use *timm.list_models()* method.

# Usage

Unlike previous backdoor attacks, the triggers of this backdoor attack is not pre-defined by the attacker but generated afterwards. 
Therefore, from the viewpoint of the attacker, we first insert a backdoor into a pre-trained model; then we generate dynamic triggers for inputs. 

## Backdoor Insertion

Given a pre-trained model whose first layer is strided, we can insert a backdoor. 

The following is an example on the *vit_small_patch16_384* model. 

```
python insert_backdoor.py --model vit_small_patch16_384 --beta1 2 --beta2 0.05
```

To fine-tune a model (full-network), you can use:

```
python translearn.py --model vit_small_patch16_384 --dataset pets37 --lr 0.001 --pretrained backdoors/TIMESTAMP/vit_small_patch16_384.pth
```

You can use the *--layer-name* to specify the layers that can be fine-tuned; only layers behind the specified layer name are optimizable. 
The layer name can be obtained by *[k for k,_ in model.named_modules()]*. 
For example, to fine-tune the *vit_small_patch16_384* model at the fixed-feature setting (i.e., only fine-tune the layer after the *norm* layer), you can use:

```
python translearn.py --model vit_small_patch16_384 --dataset pets37 --lr 0.001 --pretrained backdoors/TIMESTAMP/vit_small_patch16_384.pth --layer-name norm
```

To evaluate the performance of the model, you can use:

```
python test.py --model vit_small_patch16_384 --weight logs/vit_small_patch16_384/pets37/TIMESTAMP/vit_small_patch16_384.pth --dataset pets37
```

## Trigger Generation

Given source inputs and the target label, based on the backdoored pre-trained model, we can generate triggers. 

```
python generate_triggers.py --model vit_small_patch16_384 --weight backdoors/TIMESTAMP/vit_small_patch16_384.pth --dataset pets37 --lamb 2
```

To evaluate the ASR/EASR of triggers generated, you can use:

```
python evaluate.py --model vit_small_patch16_384 --weight logs/vit_small_patch16_384/pets37/TIMESTAMP/vit_small_patch16_384.pth  --dataset pets37 --folder triggers/pets37_vit_small_patch16_384_TIMESTAMP/
```

To evaluate the $Acc_{src}$ of the attack samples, you can use:

```
python evaluate.py --source-acc --model vit_small_patch16_384 --weight benign/finetuned/model/ckpt --dataset pets37 --folder triggers/pets37_vit_small_patch16_384_TIMESTAMP/
```

## FaceNet
The following is an example:


1. Generate a backdoored facenet:
```shell
python insert_backdoor.py --model facenet --beta1 2 --beta2 1
```

2. Finetune the backdoored facenet:
```shell
python translearn.py --model facenet --dataset cfp --pretrained backdoors/path/to/facenet.pth --save-path logs/facenet
```

3. Generate the triggers:

```shell
python generate_triggers.py --model facenet --weight backdoors/path/to/facenet.pth --dataset cfp --logpath triggers/facenet --lamb 2.5
```

4. Evaluate the triggers:

```shell
python evaluate.py --model facenet --weight logs/facenet/path/to/facenet_avgpool_1a_finetuned.pth --folder triggers/facenet/cfp_facenet_path/ --dataset cfp
```

## Wav2Vec2

The following is an example:

1. Generate a backdoored model:

```shell
python insert_backdoor.py --model facebook/wav2vec2-base --source "samples/original songs/original_crafted_fragment_popular_sig_wav3.wav" --target "samples/target commands/911.wav" --logpath backdoors/wav2vec2 --audio --beta1 2 --beta2 0.01
```

2. Finetune a backdoored model:

```shell
cd wav2vec2
python wav2vec2_finetune.py --model_id ../backdoors/wav2vec2/path/to/model/dir/ --dataset_name timit_asr --ckpt_path ../logs/wav2vec2/ --model_save_path ../logs/wav2vec2/ --data_dir ../.cache/TIMIT
```

3. Generate commandsongs:

```shell
python generate_triggers.py --audio --weight ../backdoors/wav2vec2/path/to/model/dir/ --songs samples/original\ songs/ --commands samples/target\ commands/ --lr 0.01 --iters 1000 --logpath triggers --lamb 0.1
```

4. Model inference:

```shell
python wav2vec2_infer.py --dataset_name timit_asr --model_id ../logs/wav2vec2/ --with_lm_id base-lm --data_dir ../.cache/TIMIT
python wav2vec2_infer.py --wav_folder ../triggers/wav2vec2_path/advs/ --model_id ../logs/wav2vec2/ --with_lm_id base-lm
```
