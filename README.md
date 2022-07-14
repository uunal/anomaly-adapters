<p align="center">
<img src="docs\adapter-anomaly.png" />
</p>
<h2 align="center">
<span>AnomalyAdapters</span>
</h2>

<p align="center">
Parameter-efficient Multi-Anomaly Task Detection.
</p>

## Published at
DOI: https://doi.org/10.1109/ACCESS.2022.3141161

## Datasets
The firewall dataset consists of 14,277,447 logs. Three days activity in a  corporate network are simulated. We have used all log sequence except for the first day, which includes a DoS attack. We have extracted \%0.01 of the abnormal event. Most of the data in first day is predominated by DoS attack, which we omitted and edited data without changing timeline of log events, since attack focuses on only several workstations in the network. 172,135 number of normal logs and 16,902 number of anomalous logs, which consist of DoS, Port scanning, worms and unknown machine connections. This dataset was also mentioned in finding a DoS attack at [Deeplog]. This dataset is particularly simulated for IEEE Visual Analytics Science and Technology (VAST) 2011 MiniChallenge-2.

Hadoop Distributed File Systems (HDFS) dataset was first presented in mining console logs [PCA]. It consist of 11,175,629 logs gathered from Amazon EC2 nodes. A total of 10,887,379 logs are tagged normal and 288,250 logs are tagged abnormal. Dataset can be found in LogHub, which is collection of system log datasets for AI-based analytics [Loghub]. The activities in that datasets are defined by `blockID` attribute which acts collectively or as a single event.

+ [Deeplog] Du, M., Li, F., Zheng, G. and Srikumar, V., 2017, October. Deeplog: Anomaly detection and diagnosis from system logs through deep learning. In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security (pp. 1285-1298).
+ [PCA] Xu, W., Huang, L., Fox, A., Patterson, D. and Jordan, M.I., 2009, October. Detecting large-scale system problems by mining console logs. In Proceedings of the ACM SIGOPS 22nd symposium on Operating systems principles (pp. 117-132).
+ [Loghub] He, S., Zhu, J., He, P. and Lyu, M.R., 2020. Loghub: a large collection of system log datasets towards automated log analytics. arXiv preprint arXiv:2008.06448.

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