# LGN_TunedMIXGCF
## Environment Requirement

The code has been tested running under Python 3.7.6. The required packages are as follows:

- pytorch == 1.7.0
- numpy == 1.20.2
- scipy == 1.6.3
- sklearn == 0.24.1
- prettytable == 2.1.0

## Training

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). Important argument:

- `K`
  - It specifies the number of negative instances in K-pair loss. Note that when K=1 (by default), the K-pair loss will degenerate into the BPR pairwise loss.
- `n_negs`
  - It specifies the size of negative candidate set when using MixGCF.
- `ns`
  - It indicates the type of negative sample method. Here we provide two options: rns and mixgcf.

##### Random sample(rns)

```
python main.py --dataset ali --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns rns --K 1 --n_negs 1 --his_path aliLGNrnsHis --save True --out_dir '/content/drive/MyDrive/MixGCF/input/rnsmodel_ali.ckpt'

python main.py --dataset yelp2018 --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns rns --K 1 --n_negs 1 --his_path yelpLGNrnsHis --save True --out_dir '/content/drive/MyDrive/MixGCF/input/rnsmodel_yelp.ckpt'

python main.py --dataset amazon --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns rns --K 1 --n_negs 1 --his_path amazonLGNrnsHis --save True --out_dir '/content/drive/MyDrive/MixGCF/input/rnsmodel_amazon.ckpt'
```

#####  Tuned-MixGCF

```
python main.py --dataset ali --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns mixgcf --K 1 --n_negs 32 --his_path aliLGNHis --save True --out_dir '/content/drive/MyDrive/MixGCF/input/Tunedmodel_ali.ckpt'

python main.py --dataset yelp2018 --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns mixgcf --K 1 --n_negs 64 --his_path yelpLGNHis --save True --out_dir '/content/drive/MyDrive/MixGCF/input/Tunedmodel_yelp.ckpt'

python main.py --dataset amazon --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 3 --pool mean --ns mixgcf --K 1 --n_negs 16 --his_path amazonLGNHis --save True --out_dir '/content/drive/MyDrive/MixGCF/input/Tunedmodel_amazon.ckpt'

```
