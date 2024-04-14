# SpliceBERT-analysis
Additional analysis on SpliceBERT. 
The original repository is available at [SpliceBERT](https://github.com/biomed-AI/SpliceBERT).


## Benchmark

### On SpliceAI's GTEx dataset

We fine-tuned SpliceBERT on SpliceAI's GTEx dataset with [R-Drop](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) regularization for 5 times using different random seeds (model weights: [Google Drive](https://drive.google.com/file/d/1sUrsKbe0HJfLmNxqcNkmZccy835V0UFP/view?usp=sharing)). 
The average AP scores of SpliceBERT (900nt) is comparable (donor) or slightly superior (acceptor) to SpliceAI-10K, 
while the ensemble model (averaging the predictions of 5 models) underperforms that of SpliceAI-10K, 
which is likely because that SpliceBERT were fine-tuned based on the same pre-trained model and thus lack sufficient diversity.

The source codes are shared in [benchmark_spliceai-gtex](./benchmark_spliceai-gtex)

| model | receptive field size | AP (donor) | AP (acceptor) |  
| --- | --- | --- | ---- |  
SpliceBERT  | 900  | $0.8547 \pm 0.0012$  | $0.8458 \pm 0.0009$ |  
SpliceAI-10k  | 10001  | $0.8547 \pm 0.0027$  | $0.8434 \pm 0.0023$ |  
SpliceAI-2k  | 2001 | $0.8369 \pm 0.0015$  | $0.8270 \pm 0.0017$ |  
SpliceAI-400  | 401 | $0.7961 \pm 0.0020$  | $0.7873 \pm 0.0026$ |  
SpliceAI-80  | 81 | $0.5216 \pm 0.0022$  | $0.4449 \pm 0.0020$ |  


| model (ensemble) | receptive field size | AP (donor) | AP (acceptor) |  
| --- | --- | --- | ---- |  
SpliceAI-10k (ensemble)  | 10001  | $0.8735$  | $0.8644$ |  
SpliceBERT (ensemble)  | 900  | $0.8608$  | $0.8524$ |  

