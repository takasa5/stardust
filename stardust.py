"""stardust 10/21"""
import numpy as np
import cv2
import math
import sys

WHITE = (255, 255, 255)
# TODO: 1本目の線の長さを自動判定
FIRSTL = 35 * 8
# TODO: 線の太さ、円の半径を自動判定
LWEIGHT = 1
CRADIUS = 3

SGT = {"itr":0,
       "ANGS":[85, 47, 37, 29],
       "STD_D":[1.66, 1.97, 3.69, 5.88],
       "D":[1.25, 0.905, 1.75, 2.37]
      }
#itr:どこの値を参照すべきかのイテレータ、書き込んだらインクリメント
#ANGS:角度 STD_D:基準点からの距離 D:前の点との距離
SGT_ANGS = []
SGT_STD_D = []
T = [1.66, 1.97, 3.69, 5.88]

def scale_down(image):
    hight = image.shape[0]
    width = image.shape[1]
    small = cv2.resize(image, (round(width/8), round(hight/8)))
    return small

def detect_stars(image):
    """最適(？)スレッショルドを設定し、抽出した星座標のリストを返す"""
    flag = True
    thr = 100
    #make grayscale image
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    while flag:
        stars = []
        #flag = False
        ret, new = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
        #cv2.imshow("gray", new)
        
        #detect contours of stars
        det_img, contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        im = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        if im.shape[0] > 1000 or im.shape[1] > 1000:
            im = scale_down(im)
        cv2.imshow("gray", im)

        for cnt in contours:
            M = cv2.moments(cnt)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                stars.append(np.array([[cx, cy]], dtype='int32'))
            else:
                stars.append(np.array([cnt[0][0]], dtype='int32'))
        
        if len(stars) > 150 and thr <= 250:
            #print("len:",len(stars))
            thr += 10
            del new
            #print("threshold:",thr)
        else:
            #print("len:",len(stars))
            flag = False

    if len(stars) == 0:
        # TODO:0になったら直前のスレッショルドでやるようにしたい(未検証)
        print("can't detect sky")
        thr -= 10
        del new
        ret, new = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
        det_img, contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        cv2.imshow("gray", im)
        for cnt in contours:
            M = cv2.moments(cnt)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                stars.append(np.array([[cx, cy]], dtype='int32'))
            else:
                stars.append(np.array([cnt[0][0]], dtype='int32'))
        print("threashold:",thr)
        return stars
        #sys.exit()
    else:
        print("threashold:",thr)
        #print(stars)
        return stars


def on_mouse(event, x, y, flag, param):
    """マウスクリック時"""
    #左クリックで最近傍の星出力
    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse", x, y, sep=' ', end='\n')
        print(search_near_star(x, y, 0, param))

def search_near_star(x, y, i, stars):
    """(x, y)にi番目(0オリジン)に近いものを返す"""
    p = np.array([x, y])
    L = np.array([])
    for star in stars:
        L = np.append(L, np.linalg.norm(star-p))
    index = np.array(L)
    index = np.argsort(index)
    return stars[index[i]]

def draw_line(img, stars, constellation):
    C = constellation
    for star in stars:
        std = np.array([star[0][0], star[0][1]])

    for star in stars:
        #print(star)
        std = np.array([star[0][0], star[0][1]])
        i = 1
        while True:
            p1 = search_near_star(std[0], std[1], i, stars)[0]
            #print("p1", p1[0], p1[1], sep=' ')
            d1 = np.linalg.norm(std-p1)
            if d1 > FIRSTL/7: # TODO:要検証箇所
                #print("not found")
                break

            point, bector = p1, p1-std
            print(point)
            #Falseで星座が書けるかどうかをチェック
            for k in range(4):
                point, bector = trac_constellation(False, img, point, bector, std, d1, stars, C)
                C["itr"] += 1
            C["itr"] = 0

            if point is None:
                i += 1
            else:
                #SGT_ANGS.append(theta)
                SGT_STD_D.append(d1/d1)
                cv2.line(img, (std[0],std[1]), (p1[0],p1[1]), WHITE, LWEIGHT)
                cv2.circle(img, (std[0],std[1]), CRADIUS, WHITE, LWEIGHT)

                #draw_rest_sgt(img, p1, p1-std, std, d1, stars)
                point, bector = p1, p1-std
                #Trueで記述
                for k in range(4):
                    point, bector = trac_constellation(True, img, point, bector, std, d1, stars, C)
                    C["itr"] += 1
                cv2.circle(img, (point[0], point[1]), CRADIUS, WHITE, LWEIGHT)
                C["itr"] = 0
                return

def trac_constellation(write, img, bp, bec, std_p, std_d, stars, constellation):
    """(描画判断、描画先、前の座標、前ベクトル、基準点、基準距離、星座標リスト、星座dic)"""
    C = constellation
    dist, ang = C["D"][C["itr"]], C["ANGS"][C["itr"]]
    if bp is None:
        return (None, None)

    i, p, d = 0, 0, 0
    angles = []
    A = []
    points = []
    while d/std_d < dist * 0.8:
        p = search_near_star(bp[0], bp[1], i, stars)[0]
        d = np.linalg.norm(bp- p)
        i += 1
    while d/std_d < dist * 1.3:
        dot = np.dot(bec, p-bp)
        cos = dot / (d * np.linalg.norm(bec))
        if cos > 1 or cos < -1:
            p = search_near_star(bp[0], bp[1], i, stars)[0]
            d = np.linalg.norm(bp - p)
            i += 1
        else:
            rad = math.acos(cos)
            theta = rad * 180 / np.pi
            d_s = np.linalg.norm(p-std_p)/std_d
            rd = C["STD_D"][C["itr"]]
            
            if (theta > ang-2.5 and theta < ang+2.5)  and (d_s > rd*0.8 and d_s < rd*1.2): 
                A.append(theta)
                angles.append(abs(theta-ang))
                points.append([p[0], p[1]])
                
                print("in", p, theta, sep=" ")
                p = search_near_star(bp[0], bp[1], i, stars)[0]
                d = np.linalg.norm(bp - p)
                i += 1
            else:
                print("out", p, sep=" ", end=' ')
                p = search_near_star(bp[0], bp[1], i, stars)[0]
                d = np.linalg.norm(bp - p)
                i += 1
    if len(angles) == 0:
        #print("itr:", C["itr"], "angles is empty", ang)
        return (None, None)
    else:
        tp = np.array(points[np.argmin(angles)])
        print("w:", tp)
        if write is True:
            #S += 1
            #SGT_ANGS.append(A[np.argmin(angles)])
            #SGT_STD_D.append(np.linalg.norm(tp-std_p)/std_d)
            cv2.line(img, (bp[0], bp[1]), (tp[0], tp[1]), WHITE, LWEIGHT)
            cv2.circle(img, (bp[0], bp[1]), CRADIUS, WHITE, LWEIGHT)
        return (tp, tp-bp)

if __name__ == '__main__':
    IMAGE_FILE = "1614"
    img = cv2.imread(IMAGE_FILE + ".JPG") #IMG_1618
    stars = detect_stars(scale_down(img))
    img = scale_down(img)
    draw_line(img, stars, SGT)
    
    #img = scale_down(img)
    cv2.imshow("stardust", img)
    cv2.setMouseCallback("stardust", on_mouse, stars)
    print("SGT_ANGS:",SGT_ANGS)
    print("SGT_STD_D:",SGT_STD_D)
    cv2.imwrite("SGT_" + IMAGE_FILE + ".JPG", img)
    cv2.waitKey()