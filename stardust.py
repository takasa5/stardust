"""stardust 10/21"""
import numpy as np
import cv2
import math
import sys

WHITE = (255, 255, 255)

def scale_down(image):
    hight = image.shape[0]
    width = image.shape[1]
    small = cv2.resize(image, (round(width/8), round(hight/8)))
    return small

def detect_stars(image):
    flag = True
    thr = 70
    #make grayscale image
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    while flag:
        stars = []
        #flag = False
        ret, new = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
        cv2.imshow("gray", new)
        
        #detect contours of stars
        det_img, contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #before = cv2.drawContours(before, contours, -1, (0, 255, 0), 1)
        for cnt in contours:
            M = cv2.moments(cnt)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                stars.append(np.array([[cx, cy]], dtype='int32'))
            else:
                stars.append(np.array([cnt[0][0]], dtype='int32'))
        
        if len(stars) > 150:
            print("len:",len(stars))
            thr += 10
            del new
            print("threshold:",thr)
        else:
            print("len:",len(stars))
            flag = False
    if len(stars) == 0:
        print("can't detect sky")
        sys.exit()
    else:
        return stars
    #print(stars)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse", x, y, sep=' ', end='\n')
        #print(search_near_star(x, y, 0))

def search_near_star(x, y, i, stars):
    """(x, y)にi番目(0オリジン)に近いものを返す"""
    p = np.array([x, y])
    L = np.array([])
    for star in stars:
        L = np.append(L, np.linalg.norm(star-p))
    index = np.array(L)
    index = np.argsort(index)
    return stars[index[i]]

def draw_line(img, stars):
    for star in stars:
        #print(star)
        std = np.array([star[0][0], star[0][1]])
        #std = np.array([329, 344])
        i = 1
        while True:
            p1 = search_near_star(std[0], std[1], i, stars)[0]
            #print("p1", p1[0], p1[1], sep=' ')
            d1 = np.linalg.norm(std-p1)
            if d1 > 35:
                print("not found")
                break

            j = 0
            p2 = search_near_star(p1[0], p1[1], j, stars)[0]
            #print("p2", p2[0], p2[1], sep=' ')
            d2 = np.linalg.norm(p1-p2)
            """
            while d2/d1 < 1.2: #星2と星3の距離が適正になるまで探す
                j += 1
                p2 = search_near_star(p1[0], p1[1], j)[0]
                d2 = np.linalg.norm(p1-p2)
                print("p2", p2[0], p2[0], sep=' ')
            """
            while d2/d1 < 1.5: #適正範囲中の全てについて探索
                #角度を求める
                dot = np.dot(p1-std, p2-p1)
                cos = dot / (d1*d2)
                if cos > 1 or cos < -1:
                    #print(p1, p2, sep=' ')
                    j += 1
                    p2 = search_near_star(p1[0], p1[1], j, stars)[0]
                    d2 = np.linalg.norm(p1-p2)
                    #print("p2", p2[0], p2[1], sep=' ')
                else:
                    rad = math.acos(cos)
                    theta = rad * 180 / np.pi
                    if theta > 84 and theta < 85:
                        print(theta)
                        cv2.circle(img, (std[0],std[1]), 3, WHITE, 1) #debug
                        point, bector = trac_constellation(img, 0.905, 47, p2, p2-p1, d1, stars)
                        if point is None:
                            break
                        
                        cv2.line(img, (std[0],std[1]), (p1[0],p1[1]), WHITE, 1)
                        cv2.circle(img, (std[0],std[1]), 3, WHITE, 1)
                        cv2.line(img, (p1[0],p1[1]), (p2[0],p2[1]), WHITE, 1)
                        cv2.circle(img, (p1[0], p1[1]), 3, WHITE, 1)
                        
                        #draw_rest_sgt(img, p2, p2-p1, d1)
                        draw_rest_sgt(img, point, bector, d1, stars)
                        return
                    else:
                        j += 1
                        p2 = search_near_star(p1[0], p1[1], j, stars)[0]
                        d2 = np.linalg.norm(p1-p2)
                        #print("p2", p2[0], p2[1], sep=' ')
            i += 1

def draw_rest_sgt(img, bp, bec, std_d, stars):
    """直前の点、直前のベクトル、基準距離を渡して残りを描画"""
    #point, bector = trac_constellation(img, 0.905, 47, bp, bec, std_d)
    point, bector = trac_constellation(img, 1.75, 39, bp, bec, std_d, stars)
    point, bector = trac_constellation(img, 2.05, 25, point, bector, std_d, stars)
    """
    i, p, d = 0, 0, 0
    while d/std_d < 1.9:
        p = search_near_star(point[0], point[1], i)[0]
        d = np.linalg.norm(point - p)
        i += 1
    while d/std_d < 2.2:
        #角度を計算
        dot = np.dot(bector, p-point)
        cos = dot / (d * np.linalg.norm(bector))
        if cos > 1 or cos < -1:
            cos = 1
        rad = math.acos(cos)
        theta = rad * 180 / np.pi
        if theta < 60:
            print(theta)
            cv2.line(img, (point[0],point[1]), (p[0],p[1]), WHITE, 1)
            return True
        else:
            p = search_near_star(point[0], point[1], i)[0]
            d = np.linalg.norm(point - p)
            i += 1
    
    return False
    """

def trac_constellation(img, dist, ang, bp, bec, std_d, stars):
    """(描画先、星間距離、前ベクトルとの角度、前の座標、前ベクトル、基準距離)"""
    if bp is None:
        return (None, None)
    
    i, p, d = 0, 0, 0
    while d/std_d < dist * 0.8:
        p = search_near_star(bp[0], bp[1], i, stars)[0]
        d = np.linalg.norm(bp- p)
        i += 1
    while d/std_d < dist * 1.2:
        dot = np.dot(bec, p-bp)
        cos = dot / (d * np.linalg.norm(bec))
        if cos > 1 or cos < -1:
            cos = 1
        rad = math.acos(cos)
        theta = rad * 180 / np.pi
        if theta > ang-0.5 and theta < ang+1.5:
            print(theta)
            cv2.line(img, (bp[0],bp[1]), (p[0],p[1]), WHITE, 1)
            cv2.circle(img, (bp[0],bp[1]), 3, WHITE, 1)
            return (p, p-bp)
        else:
            p = search_near_star(bp[0], bp[1], i, stars)[0]
            d = np.linalg.norm(bp - p)
            i += 1
    
    return (None, None)

if __name__ == '__main__':
    img = cv2.imread("test.JPG") #IMG_1618
    before = scale_down(img)
    stars = detect_stars(before)
    draw_line(before, stars)

    cv2.imshow("stardust", before)
    cv2.setMouseCallback("before", on_mouse)
    
    
    cv2.waitKey()