# python3 ./train_dist.py 512 AWGN 10 CCVVCCCC CCVVCCCC_10dB 1200 --filters 256 256 256 256 256 256 256 256 --repetitions 1 1 1 1 1 1 1 1 --papr None

# # note: ignore the CCVVCCC thing
# python3 ./train_dist.py 512 AWGN 10 CCVVCCCC DeepJSCC_10dB_512 1200 --filters 256 256 256 256 256 256 256 256 --repetitions 1 1 1 1 1 1 1 1 --papr None

# python3 ./train_dist.py 512 AWGN 10 CCVVCCCC DeepJSCC_10dB_noGDN 300 --filters 256 256 256 256 256 256 256 256 --repetitions 1 1 1 1 1 1 1 1 --papr None

# python3 ./train_dist.py 512 AWGN 10 CCVVCCCC DeepJSCC_10dB_noGDN-ReLU 300 --filters 256 256 256 256 256 256 256 256 --repetitions 1 1 1 1 1 1 1 1 --papr None

python3 ./train_dist.py 512 AWGN 10 CCVVCCCC CCVVCCCC_10dB 1200 --filters 256 256 256 256 256 256 256 256 --repetitions 1 1 1 1 1 1 1 1 --papr None
