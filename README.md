# Action-RecognitionVLM
Create venv
```
python -m venv venv
```
On windows
```
venv\Scripts\activate
```

On mac
```
source venv/bin/activate
```

Install the CLIP
```
pip install git+https://github.com/openai/CLIP.git
```

Activate GPU Pytorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```