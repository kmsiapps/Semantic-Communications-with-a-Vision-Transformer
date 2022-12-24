# python3 ./params_and_flops.py 512 AWGN 10 CCCCCC CCCCCC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./params_and_flops.py 512 AWGN 10 CCCVCC CCCVCC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
python3 ./params_and_flops.py 512 AWGN 10 CCCVVC CCCVVC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true

# python3 ./params_and_flops.py 512 AWGN 10 CCVCCC CCVCCC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./params_and_flops.py 512 AWGN 10 CVVCCC CVVCCC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
# python3 ./params_and_flops.py 512 AWGN 10 VVVCCC VVVCCC 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true

# python3 ./params_and_flops.py 512 AWGN 10 CCVVCC CCVVCC_NOGDN 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn false
# python3 ./params_and_flops.py 512 AWGN 10 CCVVCC CCVVCC_GDN 300 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 1 --gdn true
