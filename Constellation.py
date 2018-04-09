
#ANGS:前ベクトルからの角度 D:前のベクトルとの長さの比
#JCT:分岐点 BP:戻る点の情報、実行時に追加される REST:残りの部分、JCTと対応
#N:主に計算や判定に用いる(いるのか？) MAX:星の総数-1(最初)
#注意:基準線を長めにとると失敗しがち
class Sagittarius:
    def __init__(self):
        self.SGT5 = {"ANGS":[-48.5], "D":[1.237],
                     "JCT":[], "BP":[], "REST":[], "N":4
                    }
        self.SGT4 = {"ANGS":[+123.1, +27.85, -37.55],
                     "D":[0.8143, 0.9881, 0.59727],
                     "JCT":[0], "BP":[], "REST":[self.SGT5], "N":4
                    }
        self.SGT3 = {"ANGS":[-128, +15.17],
                     "D":[0.7564, 2.0678],
                     "JCT":[], "BP":[], "REST":[], "N":4
                    }
        self.SGT2 = {"ANGS":[-65.5, +96.3],#64.5
                     "D":[0.8987, 0.3662],
                     "JCT":[0], "BP":[], "REST":[self.SGT3], "N":4
                    }
        self.SGT = {"ANGS":[+85, +47, -37, -29],
                    "D":[1.25, 0.724, 1.934, 1.354],
                    "JCT":[0, 2], "BP":[], "REST":[self.SGT2, self.SGT4], "N":4,
                    "MAX": 13
                   }
        self.line = self.SGT
        self.ja_name = "いて座"
        self.en_name = "Sagittarius"
        self.short_name = "SGT"
sgt = Sagittarius()

class Scorpius:
    def __init__(self):
        self.SCO3 = {"ANGS":[-160.489],
                     "D":[1.1615],
                     "JCT":[], "BP":[], "REST":[], "N":5
                    }
        self.SCO2 = {"ANGS":[-157.045, 70.910, 13.447, 50.3314, 75.201, 1.771],
                     "D":[1.3837, 0.7800, 1.3409, 0.7404, 0.4268, 1.08],#最後データにばらつきあり
                     "JCT":[], "BP":[], "REST":[], "N":5
                    }
        self.SCO = {"ANGS":[22.302, 13.825, 25.867, -5.133, -80.837],
                    "D":[1.8050, 0.3505, 0.8718, 2.9250, 0.5277],
                    "JCT":[-1, 3], "BP":[], "REST":[self.SCO2, self.SCO3], "N":5,
                    "MAX":13
                   }
        self.line = self.SCO
        self.ja_name = "さそり座"
        self.en_name = "Scorpius"
        self.short_name = "SCO"
sco = Scorpius()
"""
class Perseus:
    def __init__(self):
        self.PRS3 = {"itr":0,
                     "ANGS":[122.2, 14.5, 27.9, 33.7, 40.3, 17.5, 88.2],
                     "STD_D":[4.6, 4.32, 3.59, 4.33, 6.05, 7.68, 7.64],
                     "D":[2.49, 1.1, 1.78, 1.44, 1.74, 1.65, 0.87],
                     "JCT":[], "BP":[], "REST":[], "N":10
                    }
        self.PRS2 = {"itr":0,
                     "ANGS":[60.3, 70.9],
                     "STD_D":[5.3, 9.4],
                     "D":[1.35, 4.75],
                     "JCT":[], "BP":[], "REST":[], "N":10
                    }
        self.PRS = {"itr":0,
                    "ANGS":[90.9, 37, 22.3, 21.9, 13.3, 73.9, 38.8, 52, 40.2, 66.3],
                    "STD_D":[1.13, 1.98, 2.27, 2.85, 4.19, 4.75, 5.49, 5.7, 6.58, 7.49],
                    "D":[0.5, 1.8, 0.48, 1.08, 1.96, 0.7, 1.62, 3.78, 0.88, 1.18],
                    "JCT":[4, 6], "BP":[], "REST":[self.PRS2, self.PRS3], "N":10
                   }
    def get(self):
        return self.PRS

    def get_name(self):
        return "PRS"
"""