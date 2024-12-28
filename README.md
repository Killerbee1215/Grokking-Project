# Grokking-Project
Final project of Mathematical Introduction to Machine Learning

## Environment Setup
```
conda create -n grokking python=3.9
conda activate grokking
[Conda] conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
[Pip] pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
(Alternatively, install a different version of PyTorch according to your settings.)
pip install blobfile numpy tqdm scipy mod matplotlib
pip install pytorch_lightning==1.5.0
```
## Subtask 1
```
python scripts/train.py --max_steps 100000
python scripts/visualize_metrics
```


