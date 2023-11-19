First download the data in data/ folder from https://github.com/twitter-research/tgn. We follow the similar data format as the original tgn code. 


Then run the following 
```
python tgn_ss_ance.py --gpu=1 --er=1 --n_epoch=100 --warmup_epoch=20 --num_random_samples=1 --num_hard_samples=1 --data=wikipedia --topk=5
```

All of these parameters can be varied to get the optimal results.
num_random_samples:No. of negative samples to be selected uniformly
num_hard_samples: No. of negatives samples to be selected using proposed method.
topk= No. of items to keep as per eq. 17 in paper, we use 5 in case of wiki and 50 in case of reddit