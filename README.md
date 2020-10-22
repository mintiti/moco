## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
Forked from the [original MoCo PyTorch repo](https://github.com/facebookresearch/moco).  
The code is minimal modification on the original code, with small modifications for our experiments.  
The project is made in the context of the [TissueNet: Detect Lesions in Cervical Biopsies](https://www.drivendata.org/competitions/67/competition-cervical-biopsy/page/254/)
competition. 

The idea is to adapt the MoCo method to the specificities of biopsy scans, with the goal of detecting cancers.  

Multiple changes are to be tested :  
[] Change in the pretext task used to train the MoCo encoder, and experiment with different contrastive losses  
[] Change in the key and query representation to give help the key and query have more semantics  
[] Experiment with different transformations that might be more suited to our task
[] Experiment with MoCo v2 data augmentations

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py -a resnet34 --lr 0.03 --batch-size 32 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 NVIDIA V100 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v1<br/>top-1 acc.</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">53 hours</td>
<td align="center">60.8&plusmn;0.2</td>
<td align="center">67.5&plusmn;0.1</td>
</tr>
</tbody></table>

Here we run 5 trials (of pre-training and linear classification) and report mean&plusmn;std: the 5 results of MoCo v1 are {60.6, 60.6, 60.7, 60.9, 61.1}, and of MoCo v2 are {67.7, 67.6, 67.4, 67.6, 67.3}.



# TODO :
[] Change the normalization to dataset-relevant means and standard deviations  
[] Properly test the model with validation data  
[] Compare the pretrained data classifier to a purely supervised classifier  

