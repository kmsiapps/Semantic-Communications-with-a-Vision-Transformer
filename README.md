# On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation

Code repository for the paper [On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation](https://ieeexplore.ieee.org/document/10171356).
```
@ARTICLE{10171356,
  author={Yoo, Hanju and Dai, Linglong and Kim, Songkuk and Chae, Chan-Byoung},
  journal={IEEE Access}, 
  title={On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation}, 
  year={2023},
  volume={11},
  number={},
  pages={71528-71541},
  doi={10.1109/ACCESS.2023.3291405}}
```

### Directories
- `analysis/`: Codes for various analysis, e.g., # parameters, cosine similarity, or Fourier analysis, as presented in the paper.
- `models/`: Tensorflow implementations of the proposed model (and also the baseline).
- `usrp/`: Neural network or socket-related codes related to wireless semantic communications prototype. You may also want to see [LabVIEW codes](https://github.com/kmsiapps/Semantic-Communications-with-a-Vision-Transformer/releases/tag/USRP) for USRP implementations.  

### Training
`python3 ./train_dist.py #_CONSTELLATIONS CHANNEL SNR_DB ARCH EXPERIMENT_NAME EPOCHS --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn WITH_GDN?`

e.g., `python3 ./train_dist.py 512 AWGN 15 CCVVCC CCVVCC_15dB 600 --filters 256 256 256 256 256 256 --repetitions 1 1 3 3 1 1 --gpu 0 --gdn false`

### Instructions to run USRP-based prototype
- Set up client/server address in `config/usrp_config.py`. You can either use localhost (to execute the code in the local PC) or the remote server address (some kind of GPU server). 
- Run `python ./usrp/server.py` in the tensorflow-installed PC (e.g., GPU server).
- Run `python ./usrp/client.py`. Note that it does not require tensorflow and GPU.
- In the USRP device, open attached [LabVIEW codes](https://github.com/kmsiapps/Semantic-Communications-with-a-Vision-Transformer/releases/tag/USRP).
- In the TCP settings tab, set up the client address (for the USRP to connect on). `client.py` works as a TCP server (to receive USRP connection) and a TCP client (to connect GPU server that conducts neural encoding/decoding). 
- Adjust the detection threshold in the right panel. The USRP sends the received signal to the client if the max received power is higher than the threshold. About +5 dB over the average noise level (power level without any transmission) will do. Do not use too low value for it; random noises may be transmitted to client and the entire process may be screwed up.
- Set Tx start trigger to "Rx start trigger" in the panel.
- Press the run button in the LabVIEW.
- All set! You can see a prompt (to choose an image to transmit) in the client.py if everything went well.

### Some results
![image_examples](https://user-images.githubusercontent.com/23615360/213404386-df8c94ea-0a4a-4b82-a764-8418fc67d2e0.png)
