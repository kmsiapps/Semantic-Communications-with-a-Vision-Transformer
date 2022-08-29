# python3 ./train.py 512 None 10 CCVVCCC CCVVCCC_noiseless 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None --initial_epoch 100 --ckpt CCVVCCC_noiseless_100.ckpt
# python3 ./train.py 512 AWGN 10 CCVVCCC CCVVCCC_10dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 10 CCCCCCC CCCCCCC_noiseless 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 10 CCCCCCC CCCCCCC_10dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None

# python3 ./train.py 512 AWGN 0 CCCCCCC CCCCCCC_0dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCVVCCC CCVVCCC_0dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None

# python3 ./train.py 512 AWGN 10 CCVVCCC CCVVCCC_ENC+ 100 --filters 64 96 192 384 128 64 32 --repetitions 4 4 4 4 4 2 2 --papr None
# python3 ./train.py 512 AWGN 10 CCVVCCC CCVVCCC_DEC+ 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 8 4 4 --papr None
# python3 ./train.py 512 AWGN 10 CCVVCCC CCVVCCC_ALL+ 100 --filters 64 96 192 384 128 64 32 --repetitions 3 3 3 3 6 3 3 --papr None

# python3 ./train.py 512 None 0 CCCCCCC CCCCCCC_noiseless 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 10 CCCCCVC CCCCCVC_10dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None

# python3 ./train.py 512 AWGN 0 CCCCCVC CCCCCVC_10dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None

# python3 ./train.py 512 AWGN 0 CCCVCCC CCCVCCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CVVVCCC CVVVCCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCCCVCC CCCCVCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCCVVCC CCCVVCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 0 CCCVVCC CCCVVCC_noiseless 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCVVVCC CCVVVCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 VVVVCCC VVVVCCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 10 VVVVCCC VVVVCCC_10dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 10 CCVVVVC CCVVVVC_10dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCVVVVC CCVVVVC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCVVVVV CCVVVVV_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 10 CCVVVVV CCVVVVV_10dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCCCVVC CCCCVVC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 0 CCCCVVC CCCCVVC_noiseless 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCCCVVV CCCCVVV_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 0 CCCCVVV CCCCVVV_noiseless 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CVVVVCC CVVVVCC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 0 CVVVVCC CVVVVCC_noiseless 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CVVVVVC CVVVVVC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 0 CVVVVVC CVVVVVC_noiseless 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCVVCVC CCVVCVC_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 None 0 CCVVCVC CCVVCVC_noiseless 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None

# python3 ./train.py 512 AWGN 0 CCVVCCC CCVVCCC_0dB 1200 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None --initial_epoch 300 --ckpt ./bkup_ckpt/CCVVCCC_0dB_288.ckpt

# python3 ./train.py 512 AWGN 0 CCVVCCC CCVVCCC_k=5_0dB 100 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
# python3 ./train.py 512 AWGN 0 CCVVCCC CCVVCCC_k=5_0dB 1200 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None --initial_epoch 600 --ckpt ./ckpt/CCVVCCC_k=5_0dB_587.ckpt

# python3 ./train.py 512 AWGN 10 CCVVCCC CCVVCCC_k=5_norot_10dB 1200 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None

python3 ./train.py 512 AWGN 10 CCVVCCC CCVVCCC_k=9_5_norot_10dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None
python3 ./train.py 512 AWGN 10 CCCCCCC CCCCCCC_k=9_5_norot_10dB 300 --filters 64 96 192 384 128 64 32 --repetitions 2 2 2 2 4 2 2 --papr None