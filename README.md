<p align="center">
<img src="docs\adapter-anomaly.png" />
</p>
<h2 align="center">
<span>AnomalyAdapters</span>
</h2>

<p align="center">
Parameter-efficient Multi-Anomaly Task Detection.
This work is under review.
</p>

## Testing
Latest test are done with `adapter-transformers@V2.2.0a1` 

Follow the installation at:

```https://github.com/Adapter-Hub/adapter-transformers#installation```

To use mixed precision:

CUDA and C++ extensions via
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Apex also supports a Python-only build (required with Pytorch 0.4) via
```
pip install -v --disable-pip-version-check --no-cache-dir ./
```

## Special Thanks to
[Huggingface](https://github.com/huggingface/transformers), [adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers) and [Allennlp](https://github.com/allenai/allennlp) repositories

## Logo
Just for fun, I created my version :)