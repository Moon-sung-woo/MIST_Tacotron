# GST Tacotron2 KOREAN

## DATA
### 0. Code reference
 * [NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) ,  [jinhan](https://github.com/jinhan/tacotron2-gst)의 코드를 참고하여 만들었습니다.
### 1. Dataset 
  * [Korean Speech Emotion Dataset](http://aicompanion.or.kr/kor/main/)
  * Single Female Voice Actor recorded six diffrent emotions(neutral, happy, sad, angry, disgust, fearful), each with 3,000 sentences. Total 30 hours

### 2. Text
 * Using [korean_cleaner](https://github.com/Yeongtae/tacotron2/tree/master/text)
 * Using jamo
  ```
    안녕하세요.
     ==>
     ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ 
   ```

### 3. Audio
* sampling rate: 16000
* filter length: 1024
* hop length: 256
* win length: 1024
* n_mel: 80
* mel_fmin: 0
* mel_fmax: 8000

### 4. file list
  * path | text
 ```
dataset/hap/wav/acriil_hap_00003104.wav|경암은 푸른 수풀 속에 거뭇거뭇 보이는 높은 기와집들을 손가락질로 가리키며 자랑스런 얼굴로 무어라고 중얼거렸다.
dataset/neu/wav/acriil_neu_00000097.wav|모든 것을 공개할 수 없으나 앞으로 국민화합과 화해조치들을 강구해 나갈 것이다.
dataset/fea/wav/acriil_fea_00002629.wav|우리집 개와 고양이는 사이가 좋다.
 ```
 
## Requirement
```
torch = 1.6.0
librosa = 0.8.0
```

## How to use

### training
 1) Download [Dataset](http://aicompanion.or.kr/kor/main/)
 
 2) Make path like (dataset/fea/wav/acriil_fea_00002629.wav)
 
 3) Make raw file to wav file
  ```
  python raw2wav.py
  ```
  
 4) Preprecess audio
  ```
  python preprocess_audio.py -f [filelist name]
  ```
  
  5) GST Tacotron train
  ```
  python -m multiproc train.py -m Tacotron2 -o ./output/ -lr 1e-3 --epochs 1501 -bs 16 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.1
  ```
  
  6) Wave Glow train
  ```
  python -m multiproc train.py -m WaveGlow -o ./output/ -lr 1e-4 --epochs 1501 -bs 4 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json
  ```

### inference
1) Write the sentence you want in the text.txt file.

2) Generate audio
```
python inference.py --tacotron2 <tacotron checkpoint path> --max-decoder-steps 2000 --waveglow <waveglow checkpoing path> -o <output path> --include-warmup -i text.txt --fp16 --ref_mel <reference audio path>

(example)
python inference.py --tacotron2 output/checkpoint_Tacotron2_300.pt --max-decoder-steps 2000 --waveglow output/checkpoint_WaveGlow_300.pt -o output/ --include-warmup -i text.txt --fp16 --ref_mel dataset/sur/wav/acriil_sur_00000808.wav
```
3) Check output path

## Result
![image](https://user-images.githubusercontent.com/53896208/106356992-ca6e0300-6346-11eb-8bef-85be19548d6e.png)

## Sample Audio

You can check sample Audio file from sample_audio folder in this project
