# python3 ./train_dist.py 512 AWGN 10 CCCVCC CCCVCC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1
# python3 ./train_dist.py 512 AWGN 10 CCCVVC CCCVVC 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1
# python3 ./train_dist.py 512 AWGN 10 CCCVVV CCCVVV 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1

# python3 ./train_dist.py 512 AWGN 10 CCCCCC CCCCCC 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --initial_epoch 100 --ckpt ./CCCCCC_100
# python3 ./train_dist.py 512 AWGN 10 CCVVCC CCVVCC_GDN 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 10 CCVVCC CCVVCC_NOGDN 100 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false

# python3 ./train_dist.py 512 AWGN 0 CCVVCC CCVVCC_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 512 AWGN 2 CCVVCC CCVVCC_2dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 512 AWGN 5 CCVVCC CCVVCC_5dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false  --initial_epoch 464 --ckpt ./ckpt/CCVVCC_5dB_460

# python3 ./train_dist.py 256 AWGN 10 CCVVCC CCVVCC_256_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 256 AWGN 0 CCVVCC CCVVCC_256_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 256 AWGN 2 CCVVCC CCVVCC_256_2dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false

# python3 ./train_dist.py 1024 AWGN 10 CCVVCC CCVVCC_1024_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false --initial_epoch 317 --ckpt ./ckpt/CCVVCC_1024_10dB_311

# python3 ./train_dist.py 1024 AWGN 0 CCVVCC CCVVCC_1024_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 1536 AWGN 0 CCVVCC CCVVCC_1536_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false

# python3 ./train_dist.py 512 AWGN 0 CCCCCC CCCCCC_512_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 2 CCCCCC CCCCCC_512_2dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 5 CCCCCC CCCCCC_512_5dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 7 CCCCCC CCCCCC_512_7dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 10 CCCCCC CCCCCC_512_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 12 CCCCCC CCCCCC_512_12dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 512 AWGN 15 CCCCCC CCCCCC_512_15dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true

# python3 ./train_dist.py 768 AWGN 10 CCCCCC CCCCCC_768_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./train_dist.py 1024 AWGN 10 CCCCCC CCCCCC_1024_10dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true

# python3 ./train_dist.py 1536 AWGN 0 CCCCCC CCCCCC_1536_0dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true

# python3 ./train_dist.py 512 AWGN 10 CCCCCC CCCCCC_512_10dB_II 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true

# python3 ./train_dist.py 512 Rayleigh 5 CCVVCC CCVVCC_512_5dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 512 Rayleigh 7 CCVVCC CCVVCC_512_7dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 512 Rayleigh 10 CCVVCC CCVVCC_512_10dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false --ckpt ./ckpt/CCVVCC_512_10dB_Rayleigh_99 --initial_epoch 118

# python3 ./train_dist.py 512 Rayleigh 12 CCVVCC CCVVCC_512_12dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 
# python3 ./train_dist.py 512 Rayleigh 15 CCVVCC CCVVCC_512_15dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 
# python3 ./train_dist.py 512 Rayleigh 0 CCVVCC CCVVCC_512_0dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 

# python3 ./train_dist.py 512 Rician 0 CCVVCC CCVVCC_512_0dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 
# python3 ./train_dist.py 512 Rician 2 CCVVCC CCVVCC_512_2dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 
# python3 ./train_dist.py 512 Rician 5 CCVVCC CCVVCC_512_5dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false

# python3 ./train_dist.py 512 Rician 7 CCCCCC CCCCCC_512_7dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 512 Rician 10 CCCCCC CCCCCC_512_10dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./train_dist.py 512 Rician 12 CCCCCC CCCCCC_512_12dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 
# python3 ./train_dist.py 512 Rician 15 CCCCCC CCCCCC_512_15dB_Rician 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false 

python3 ./train_dist.py 512 Rayleigh 0 CCCCCC CCCCCC_512_0dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true 
python3 ./train_dist.py 512 Rayleigh 2 CCCCCC CCCCCC_512_2dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true 
python3 ./train_dist.py 512 Rayleigh 5 CCCCCC CCCCCC_512_5dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
python3 ./train_dist.py 512 Rayleigh 7 CCCCCC CCCCCC_512_7dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true 
python3 ./train_dist.py 512 Rayleigh 10 CCCCCC CCCCCC_512_10dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true 
python3 ./train_dist.py 512 Rayleigh 12 CCCCCC CCCCCC_512_12dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
python3 ./train_dist.py 512 Rayleigh 15 CCCCCC CCCCCC_512_15dB_Rayleigh 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
