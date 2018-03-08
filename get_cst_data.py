import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":
    
    # 星座線データから目的星座のデータを取り出す
    tgt_cst = []
    with open("line.csv", newline="") as f:
        data = csv.reader(f)
        for line in data:
            if line[0] == "1":
                for i, e in enumerate(line):
                    line[i] = float(e) 
                tgt_cst.append(line)
    
    # 星座番号を削除
    for d in tgt_cst:
        del d[0]
    
    x_list, y_list = [], []
    print(tgt_cst)
    for d in tgt_cst:
        for e in d:
            e = np.deg2rad(e)
        print(d)
        h = np.arcsin(np.sin(d[1])*np.sin(np.deg2rad(22))+np.cos(d[1])*np.cos(np.deg2rad(22))*np.cos(d[0]))
        real_h = 100 * np.sin(h)
        A = np.arcsin((np.cos(d[1])*np.sin(d[0]))/np.cos(h))
        
        x_list.append(np.rad2deg(A))
        y_list.append(real_h)
    print(x_list)
    print(y_list)
    plt.scatter(x_list, y_list)
    plt.show()    