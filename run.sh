# Note: ignore the CCVVCCCC thing

# python3 ./train_dist.py 512 AWGN 10 CCCCCC CCCCCC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0
# python3 ./train_dist.py 512 AWGN 10 CCVCCC CCVCCC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0
# python3 ./train_dist.py 512 AWGN 10 CVVCCC CVVCCC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0
# python3 ./train_dist.py 512 AWGN 10 VVVCCC VVVCCC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0

# python3 ./train_dist.py 512 AWGN 10 CCVVCC CCVVCC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0
# python3 ./train_dist.py 512 AWGN 10 CCVVCC CCVVCC 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --initial_epoch 100 --ckpt ./CCVVCC_100

# python3 ./train_dist.py 512 AWGN 15 CCVVCC CCVVCC_15dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false
# python3 ./train_dist.py 512 AWGN 12 CCVVCC CCVVCC_12dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false
# python3 ./train_dist.py 512 AWGN 10 CCVVCC CCVVCC_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false --initial_epoch 444 --ckpt ./ckpt/CCVVCC_10dB_433
# python3 ./train_dist.py 512 AWGN 7 CCVVCC CCVVCC_7dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false

# python3 ./train_dist.py 256 AWGN 7 CCVVCC CCVVCC_256_7dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false
# python3 ./train_dist.py 256 AWGN 12 CCVVCC CCVVCC_256_12dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false
# python3 ./train_dist.py 256 AWGN 5 CCVVCC CCVVCC_256_5dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false

# python3 ./train_dist.py 768 AWGN 10 CCVVCC CCVVCC_768_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false --initial_epoch 387 --ckpt ./ckpt/CCVVCC_768_10dB_387
# python3 ./train_dist.py 1536 AWGN 10 CCVVCC CCVVCC_1536_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false

# python3 ./train_dist.py 768 AWGN 0 CCVVCC CCVVCC_768_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false

python3 ./train_dist.py 768 AWGN 0 CCCCCC CCCCCC_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false