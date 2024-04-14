# Benchmarking SpliceBERT on SpliceAI's dataset

## Data availability weights

- Pre-trained model: [zenodo](https://zenodo.org/records/7995778)  
- Fine-tuned model on GTEx-SpliceAI dataset: [Google Drive](https://drive.google.com/file/d/1sUrsKbe0HJfLmNxqcNkmZccy835V0UFP/view?usp=sharing)  
- The gtex-dataset (`gtex_dataset.txt`): https://basespace.illumina.com/s/5u6ThOblecrh (registration required).

## Fine-tuning

- Run `train_splicebert.py`

```bash 
#!/bin/bash

HG19_FASTA_H5="hg19.fa.lzf.h5" # download from Google Drive(https://drive.google.com/file/d/1amJJRtMKrgnrADwi7bgrK3aTMtV_LOAq/view?usp=drive_link) or convert from fasta file using fasta2hdf5.py (https://gist.github.com/chenkenbio/fb95823fa2dce71aee048973270473e0)

for lr in 1e-4; do
    for a in 10; do
        for seed in `seq 2020 2024`; do
            output=finetune_SpliceBERT_GTEx_rdrop.a$a.lr$lr.seed$seed
            test -e ${output}.log && continue
            ./train_splicebert.py \
                --genome $HG19_FASTA_H5 \
                -a $a \
                --lr $lr \
                --seed $seed \
                -o $output &> ${output}.log
        done
    done
done
```
