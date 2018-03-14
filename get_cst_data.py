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
    STD, STDP = 1, None
    b_bec = None
    written_list = []
    ANGS, D, STD_D = [], [], []
    for d in tgt_cst:
        x1, x2 = -d[0] * ZOOM, -d[2] * ZOOM
        y1, y2 = d[1] * ZOOM, d[3] * ZOOM
        #メルカトル図法
        #x1 += 300
        #x2 += 300
        #y1 = np.log(np.tan(np.pi/4 + np.deg2rad(d[1])/2))
        #y2 = np.log(np.tan(np.pi/4 + np.deg2rad(d[3])/2))
        
        #正距方位図法
        #x1 = - np.cos(np.deg2rad(d[0] - 270)) * (90 - d[1])
        #x2 = - np.cos(np.deg2rad(d[2] - 270)) * (90 - d[3])
        #y1 = np.sin(np.deg2rad(d[0] - 270)) * (90 - d[1])
        #y2 = np.sin(np.deg2rad(d[2] - 270)) * (90 - d[3])
        if STDP is None:
            STDP = np.array([x1, y1])        
        bec1 = np.array([x1, y1])
        bec2 = np.array([x2, y2])
        n_bec = bec2 - bec1
        dist = np.linalg.norm(n_bec)
        dist = round(dist/STD, 2)
        if b_bec is not None:
            cos = np.dot(b_bec, n_bec) / (np.linalg.norm(b_bec)*np.linalg.norm(n_bec))
            rad = np.arccos(cos)
            ang = round(np.rad2deg(rad), 1)
            y1_buf = y1
            while y1_buf in written_list:
                y1_buf = y1_buf + 1
            plt.text(x1, y1_buf, str(ang), color='g')
            written_list.append(y1_buf)
            std_d = np.linalg.norm(bec2 - STDP) / STD
            std_d = round(std_d, 2)

            STD_D.append(std_d)
            ANGS.append(ang)
            D.append(dist)
        
        plt.text((x1+x2)/2, (y1+y2)/2, str(dist), color='r')
        

        x_list.append([x1, x2])
        y_list.append([y1, y2])
        b_bec = n_bec
        plt.plot([x1, x2], [y1, y2], "-k")
    
    ret_dict = {"ANGS":ANGS,
                "D":D,
                "STD_D":STD_D
               }
    print(ret_dict)
    plt.scatter(x_list, y_list)
    plt.axis("equal")
    plt.show()    
    