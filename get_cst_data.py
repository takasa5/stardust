import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":
    # 星座線データから目的星座のデータを取り出す
    tgt_cst = []
    with open("line.csv", newline="") as f:
        data = csv.reader(f)
        for line in data:
            if line[0] == "78":
                for i, e in enumerate(line):
                    line[i] = float(e) 
                tgt_cst.append(line)
    
    # 星座番号を削除
    for d in tgt_cst:
        del d[0]
    
    x_list, y_list = [], []
    ZOOM = 1
    STD = 2.46
    b_bec = None
    written_list = []
    print(tgt_cst)
    for d in tgt_cst:
        x1, x2 = -d[0] * ZOOM, -d[2] * ZOOM
        y1, y2 = d[1] * ZOOM, d[3] * ZOOM
        bec1 = np.array([x1, y1])
        bec2 = np.array([x2, y2])
        n_bec = bec2 - bec1
        dist = np.linalg.norm(n_bec)
        if b_bec is not None:
            cos = np.dot(b_bec, n_bec) / (np.linalg.norm(b_bec)*np.linalg.norm(n_bec))
            rad = np.arccos(cos)
            ang = np.rad2deg(rad)
            y1_buf = y1
            while y1_buf in written_list:
                y1_buf = y1_buf + 1.0
            plt.text(x1, y1_buf, str(round(ang, 1)), color='g')
            written_list.append(y1_buf)

        plt.text((x1+x2)/2, (y1+y2)/2, str(round(dist/STD, 2)), color='r')
        
        x_list.append([x1, x2])
        y_list.append([y1, y2])
        b_bec = n_bec
        plt.plot([x1, x2], [y1, y2], "-k")
        
    plt.scatter(x_list, y_list)
    plt.show()    