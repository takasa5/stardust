# stardust
Detect constellation from starry pictures and draw constellation line.  
2017/10~

## Example
stardust can detect constellation from astronomical pictures(Photo by Rikuo Uchiki).

|input|output|
|---|---|
|<img src="./example_input.JPG" width=400px>|<img src="./example_output.JPG" width=400px>| 
 
stardust can also detect constellation from pictures that include ground(Photo by takasa5).

|input|output|
|---|---|
|<img src="./example_input2.JPG" width=400px>|<img src="./example_output2.JPG" width=400px>| 

## Background and Goal
When my first time of watching starry sky I tried to find some constellations, but I could find few prominent constellations.  
Typically, without prior knowledge about position of constellations, finding them is so difficult.  
Now, with a smartphone app, we can know constellation's approximate position (thanks to GPS & gyro sensor).  
So, my goal is to identify constellation and it's position **accurately** from image.  
Combining this system and GPS and gyro (and high sensitively camera), we can starry sky with constellation information.


## Requirements
```
opencv-python==3.4.0
numpy==1.14.0
```  

## Usage
```
$ git clone https://github.com/takasa5/stardust
$ cd stardust
```
```python
from stardust import Stardust
import Constellation as cs

# Initialize Stardust with image path (or image array)
sd = Stardust("./input.jpg")
# Select constellation from Constellation-class
cstl = cs.Sagittarius()
# Draw constellation line if detected
sd.draw_line(cstl)
# Get image 
ret = sd.get_image()
# cv2.imshow() or cv2.imwrite()
```

## TODO
- [ ] 精度向上
    - [x] 尤度計算式の改善
    - [ ] 星座間位置関係の考慮
    - [x] 同じ星を二度以上通る場合の考慮
- [ ] 星座データの追加
    - [ ] おひつじ座(Aries)
    - [x] おうし座(Taurus)
    - [x] ふたご座(Gemini)
    - [ ] かに座(Cancer)
    - [ ] しし座(Leo)
    - [ ] おとめ座(Virgo)
    - [ ] てんびん座(Libra)
    - [x] さそり座(Scorpius)
    - [x] いて座(Sagittarius)
    - [ ] やぎ座(Capricornus)
    - [ ] みずがめ座(Aquarius)
    - [ ] うお座(Pisces)
    - [x] オリオン座(Orion)
    - データを紹介してくれる方、撮影した写真を提供してくれる方を募集しています
- [x] 星座名の付記
    - 文字の大きさ、太さは要調整
- [ ] 画像から大まかな方角や仰角を検出 
