# stardust
星座検出プログラム  

## Example
(Photo by Rikuo Uchiki)  

|input|output|
|---|---|
|<img src="./example_input.JPG" width=400px>|<img src="./example_output.JPG" width=400px>| 
 

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
import Constellation

# 入力画像のパス(or ndarray化した画像)で初期化
sd = Stardust("./input.jpg")
# 検出したい星座を指定
cs = Constellation.Sagittarius() # いて座
# 星座線を引く(あれば)
sd.draw_line(cs.get())
# 画像を返す
ret = sd.get_image()
# cv2.imshow()なり cv2.imwrite()なりする
```

## TODO
- 精度向上
- 星座データの追加
    - 現状はいて座(Sagittarius)とさそり座(Scorpius)のみ
    - データを紹介してくれる方、撮影した写真を提供してくれる方を募集しています
- 星座名の付記
- 星座間位置関係の考慮
