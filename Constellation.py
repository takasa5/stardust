"""星座データベース : Original Constellation Database"""
#ANGS:前ベクトルからの角度 D:前のベクトルとの長さの比
#JCT:分岐点 BP:戻る点の情報、実行時に追加される REST:残りの部分、JCTと対応
#N:主に計算や判定に用いる(いるのか？) MAX:星の総数-1(最初)
#注意:基準線を長めにとると失敗しがち 「明るい、近い、近くに無関係の明るい星がない」やつを選ぶ
DEFAULT = 0
IAU = 1
class Sagittarius:
    def __init__(self):
        self.SGR5 = {"ANGS":[-48.5], "D":[1.237],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.SGR4 = {"ANGS":[+123.1, +27.85, -37.55],
                     "D":[0.8143, 0.9881, 0.59727],
                     "JCT":[0], "BP":[], "REST":[self.SGR5]
                    }
        self.SGR3 = {"ANGS":[-128, +15.17],
                     "D":[0.7564, 2.0678],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.SGR2 = {"ANGS":[-65.5, +96.3],#64.5
                     "D":[0.8987, 0.3662],
                     "JCT":[0], "BP":[], "REST":[self.SGR3]
                    }
        self.SGR = {"ANGS":[+85, +47, -37, -29],
                    "D":[1.25, 0.724, 1.934, 1.354],
                    "JCT":[0, 2], "BP":[], "REST":[self.SGR2, self.SGR4],
                    "N":4, "MAX": 13
                   }
        self.iau5 = {"ANGS":[None], "D":[7]}
        self.iau4 = {"ANGS":[None], "D":[5]}
        self.iau3 = {"ANGS":[None], "D":[7]}
        self.iau2 = {"ANGS":[None], "D":[3]}
        self.iau = {"ANGS":[+85, +47, -37, +93, -58, +124],
                    "D":[1.25, 0.724, 1.934, 1.093, 0.731, 1.625],
                    "JCT":[-2, -2, 1, 3], "BP":[],
                    "REST":[self.iau2, self.iau3, self.iau4, self.iau5],
                    "N":6, "MAX": 8 # 本当は7
                   }
        self.line = self.SGR
        self.ja_name = "いて座"
        self.en_name = "Sagittarius"
        self.short_name = "SGR"
sgr = Sagittarius()

class Scorpius:
    def __init__(self):
        """
        self.SCO3 = {"ANGS":[-160.489],
                     "D":[1.1615],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.SCO2 = {"ANGS":[-157.045, 70.910, 13.447, 50.3314, 75.201, 1.771],
                     "D":[1.3837, 0.7800, 1.3409, 0.7404, 0.4268, 1.08],#最後データにばらつきあり
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.SCO = {"ANGS":[22.302, 13.825, 25.867, -5.133, -80.837],
                    "D":[1.8050, 0.3505, 0.8718, 2.9250, 0.5277],
                    "JCT":[-2, 3], "BP":[], "REST":[self.SCO2, self.SCO3], "N":5,
                    "MAX":13
                   }
        """
        self.SCO4 = {"ANGS": [70.9, 13.4, 50.3, 75.2, 1.3, -132.0],
                     "D": [0.780, 1.341, 0.740, 0.427, 1.813, 1.220],
                     "JCT": [], "BP": [], "REST": []
                    }
        self.SCO3 = {"ANGS": [-160.5], "D": [1.162],
                     "JCT": [], "BP": [], "REST": []
                    }
        self.SCO2 = {"ANGS": [-122.1, -80.8], "D": [1.369, 0.528],
                     "JCT": [0], "BP": [], "REST": [self.SCO3]
                    }
        self.SCO = {"ANGS": [-25.9, -13.8, -22.3, -1.0],
                    "D": [1.147, 2.853, 0.554, 1.178],
                    "JCT": [-2, 3], "BP": [],
                    "REST": [self.SCO2, self.SCO4],
                    "N": 4, "MAX": 13
                   }
        self.line = self.SCO
        self.iau = self.SCO
        self.ja_name = "さそり座"
        self.en_name = "Scorpius"
        self.short_name = "SCO"
sco = Scorpius()

class Gemini:
    def __init__(self):
        self.GEM3 = {"ANGS": [83.6, -14.3, -29.5, -18.2],
                     "D": [1.460, 0.692, 0.341, 1.411],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.GEM2 = {"ANGS": [64.95, 11.5, 82.9],
                     "D": [0.736, 1.890, 0.530],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.GEM = {"ANGS": [-36.6, 75.6, 75.36, -64.5],
                    "D": [1.527, 0.545, 1.171, 1.014],
                    "JCT": [-1, 2], "BP": [], "REST": [self.GEM2, self.GEM3], "N": 4,
                    "MAX": 12
                   }
        self.iau7 = {"ANGS": [74.5], "D": [2.276],
                     "JCT":[], "REST":[], "BP":[]
                    }
        self.iau6 = {"ANGS": [-160.9, -14.3, -29.5, -18.2],
                     "D": [1.481, 0.692, 0.341, 1.411],
                     "JCT":[0], "REST":[self.iau7], "BP":[]
                    }
        self.iau5 = {"ANGS": [115.5], "D": [1.0145],
                     "JCT":[], "REST":[], "BP":[]
                    }
        self.iau4 = {"ANGS": [53.6, -61.8],
                     "D": [0.728, 1.600],
                     "JCT":[], "REST":[], "BP":[]
                    }
        self.iau3 = {"ANGS": [-144.2, -35.3, 11.5],
                     "D": [1.146, 0.6516, 1.890],
                     "JCT":[0], "REST": [self.iau4],
                     "BP": []
                    }
        self.iau2 = {"ANGS": [-68.9], "D": [0.598],
                     "JCT":[], "REST":[], "BP":[]
                    }
        self.iau = {"ANGS": [-50.6, -17.3, -123.6],
                    "D": [1.016, 1.647, 1.325],
                    "JCT": [-1, -1, 1, 1],
                    "REST": [self.iau2, self.iau3, self.iau5, self.iau6],
                    "BP":[], "N": 3,
                    "MAX": 16
                   }
        self.line = self.GEM
        self.ja_name = "ふたご座"
        self.en_name = "Gemini"
        self.short_name = "GEM"
gem = Gemini()

class Taurus():
    def __init__(self):
        self.TAU6 = {"ANGS": [140.4, -103.2, -62.25],
                     "D": [1.004, 0.264, 2.118],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.TAU5 = {"ANGS": [161.8, -13.4, 141.4, -121.5],
                     "D": [1.125, 1.743, 1.050, 0.888],
                     "JCT":[0], "BP":[], "REST":[self.TAU6]
                    }
        self.TAU4 = {"ANGS": [52.6, 78.0, -22.8, 87.6],
                     "D": [0.670, 1.491, 0.864, 3.600],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.TAU3 = {"ANGS": [-17.2], "D": [2.398],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.TAU2 = {"ANGS": [-30.1], "D": [3.172],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.TAU = {"ANGS": [-7.4, -124.2, -15.4, -2.7],
                    "D": [1.189, 0.955, 1.008, 2.405],
                    "JCT": [-2, 0, 2, 3], "BP": [], "REST": [self.TAU2, self.TAU5, self.TAU4, self.TAU3], "N": 4,
                    "MAX": 18
                   }
        self.iau3 = {"ANGS": [-109.6, 60.0],
                     "D": [0.099, 9.702],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.iau2 = {"ANGS": [161.8, -16.2, 139.0],
                     "D": [1.125, 1.589, 1.110],
                     "JCT":[1], "BP":[], "REST":[self.iau3]
                    }
        self.iau = {"ANGS": [-7.4, -124.2, -15.4, -2.7],
                    "D": [1.189, 0.955, 1.008, 2.405],
                    "JCT": [-2, 0, 3], "BP": [], "REST": [self.TAU2, self.iau2, self.TAU3], "N": 4,
                    "MAX": 10
                   }
        self.line = self.TAU
        self.ja_name = "おうし座"
        self.en_name = "Taurus"
        self.short_name = "TAU"
tau = Taurus()

class Orion():
    def __init__(self):
        self.ORI6 = {"ANGS": [None], "D": [0]}
        self.ORI5 = {"ANGS": [78.9, 24.3, 43.4],
                     "D": [0.521, 1.884, 1.221],
                     "JCT": [], "BP": [], "REST": [],
                    }
        self.ORI4 = {"ANGS": [-157.9, 109.3, -2.7, 44.3],
                     "D": [1.584, 0.163, 2.267, 0.410],
                     "JCT": [], "BP": [], "REST": [],
                    }
        self.ORI3 = {"ANGS": [114.6, -49.0, -82.2],
                     "D": [0.818, 0.636, 1.282],
                     "JCT": [0, 2, 2], "BP": [],
                     "REST": [self.ORI4, self.ORI5, self.ORI6],
                    }
        self.ORI2 = {"ANGS": [None], "D": [4]}
        self.ORI = {"ANGS": [-10.2, 104.1, 108.4],
                    "D": [0.979, 6.673, 0.921],
                    "JCT": [-2, 0], "BP": [],
                    "REST": [self.ORI2, self.ORI3],
                    "N": 3, "MAX": 10 # "砂時計"で出るように意図的に調整
                   }
        self.line = self.ORI
        self.iau = self.ORI
        self.ja_name = "オリオン座"
        self.en_name = "Orion"
        self.short_name = "ORI"
ori = Orion()

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