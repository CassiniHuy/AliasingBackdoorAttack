This is the implementation of our paper "Aliasing Backdoor Attacks on Pre-trained models" (USENIX Security '23).

I will give a tutorial and add the face/speech recognition code when I am available. 

# Installation

## Requirements

This repo is implemented by PyTorch and you should have it installed with cuda version following the guidance from https://pytorch.org/.

## Datasets

- You can find where to download the datasets in *transfer/datasets.py*. 
- Edit the config file *transfer/datasets_configs.py* to specify where the data is placed.
- You can also edit the default fine-tuning setting in *transfer/finetune_config.py* if you need.

## Models

All the image models are from the [Timm library](https://github.com/huggingface/pytorch-image-models) except the *resnet50_miil_21k* model. 

```
pip install timm
```

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
