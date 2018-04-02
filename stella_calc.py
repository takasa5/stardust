import numpy as np
import math

STD = np.array([1084, 156])
SBEC = np.array([1102, 230]) - STD
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

def new_stella_calc(bbec, nbec):
    bd = np.linalg.norm(bbec)
    nd = np.linalg.norm(nbec)
    print("D:", nd / bd)
    cos = np.dot(bbec, nbec) / (bd * nd)
    rad = np.arccos(cos)
    deg = np.rad2deg(rad)
    if np.cross(bbec, nbec) < 0:
        deg = -deg
    print("ANGS:", deg)

if __name__ == '__main__':
    a = np.array([1102, 230])
    b = np.array([1084, 156])
    c = np.array([1093, 318])
    d = np.array([778, 564])
    new_stella_calc(b-a, c-a)