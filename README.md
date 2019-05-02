# mac-network-pytorch

This is a Pytorch implementation of the 2018 ICLR Paper [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067) (MAC Network), based mostly on the [original implementation](https://github.com/stanfordnlp/mac-network) by the authors and partly (mainly question preprocessing) on the implementation by [rosinality](https://github.com/rosinality/mac-network-pytorch).

Requirements:
```
pip install -r requirements.txt
```

Prepare dataset:
- Download and extract [CLEVR v1.0 dataset](http://cs.stanford.edu/people/jcjohns/clevr/)
- Preprocess question data:  `python preprocess.py [CLEVR directory]`
- Extract image features with ResNet 101 as described in the original [Git](https://github.com/stanfordnlp/mac-network#feature-extraction)
- Put extracted features and preprocessed question data into the `data` folder so you have the following files:
    - `data/train_features.h5`
    - `data/val_features.h5`
    - `data/train.pkl`
    - `data/val.pkl`
    - `data/dic.pkl`

To train:
```
python code/main.py --cfg cfg/clevr_train_mac.yml --gpu 0
```
The basic implementation closely mirrors the parameters and config settings from the original implementation's args1.txt, i.e. this line in the original [Git](https://github.com/stanfordnlp/mac-network#model-variants): `python main.py --expName "clevrExperiment" --train --testedNum 10000 --epochs 25 --netLength 4 @configs/args.txt`

Our implementation reaches around 93-95% accuracy on the validation set after five epochs, 95-96% after ten epochs
