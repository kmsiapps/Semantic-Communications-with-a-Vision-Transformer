# End to end deep neural network based semantic communication system for image

Code repository for ICC 2022 demo proposal "Real-Time Semantic Communications with a Vision Transformer".

Notable codes:
- `utils/qam_modem_tf.py`: Tensorflow implementation of QAM modulation (Supports up to 256QAM, but easily extendable by modifying `QAMDemodulator.call()` method).
- `utils/qam_modem_naive.py`: Pure python implementation of QAM modulation.

## System architecture

![sysarch](https://user-images.githubusercontent.com/23615360/155274350-3c9cf90f-cef4-4e1c-8e88-cbb0284b1923.png)

## Results
![results-summary](https://user-images.githubusercontent.com/23615360/155274345-82ef6780-9e48-4a9f-9cff-923d68556dca.png)
