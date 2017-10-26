import numpy as np
import math

STD = np.array([341, 190])
SBEC = np.array([325, 218]) - STD
def stella_calc(std, bb, b, n):
    bbec = b - bb #前のベクトル
    nbec = n - b #今のベクトル
    d = np.linalg.norm(nbec) #前の点との距離
    db = np.linalg.norm(bbec)
    cos = np.dot(bbec, nbec) / (d * db)
    rad = math.acos(cos)
    ang = rad * 180 / np.pi #前ベクトルとの角度
    std_d = np.linalg.norm(n - std) #基準点からの距離
    ds = np.linalg.norm(SBEC)
    cos = np.dot(SBEC, nbec) / (d * ds)
    ang2 = math.acos(cos) * 180 / np.pi #基準ベクトルとの角度

    print("ANGS:",ang)
    print("STD_D:", std_d / ds)
    print("D:", d / ds)
    print("STD_A:", ang2)

if __name__ == '__main__':
    a = np.array([168, 300])
    b = np.array([157, 275])
    c = np.array([181, 293])
    stella_calc(STD, c, a, b)