"""stardust 10/21"""
import numpy as np
import cv2
import math, os, sys, time
import Constellation

END = (2**16, 2**16)
WHITE = (255, 255, 255)
LWEIGHT = 1
CRADIUS = 3
THREASH = 1000 #画像から検出したいおおよその星の数→光害除去用
SIZE = 666 #画像サイズ(横)
ARANGE = 4 #許容角度範囲(±)
DEPTH = 5 #探索近隣星数上限
STARSIZE = 120
star_count = 0

def scale_down(image, scale):
    hight = image.shape[0]
    width = image.shape[1]
    small = cv2.resize(image, (round(width/scale), round(hight/scale)))
    return small

def darken(image, gamma):
    """ガンマ補正をかける　gammma < 1で暗くなる"""
    lut = np.ones((256, 1), dtype='uint8') * 0
    for i in range(256):
        lut[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    image_gamma = cv2.LUT(image, lut)
    return image_gamma

def detect_stars(image):
    """最適(？)スレッショルドを設定し、抽出した星座標のリストを返す"""
    flag = True
    thr = 100
    gam, adapt = 1, 1
    
    #BIGMODE用処理
    global CRADIUS, LWEIGHT
    #if min(img.shape[0], img.shape[1]) > 1600:
    CRADIUS, LWEIGHT = int(max(img.shape[0], img.shape[1])/250), int(max(img.shape[0], img.shape[1])/1000)
    
    #輪郭検出用グレースケール画像生成
    astars = []
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    while flag:
        stars, areas = [], []
        ret, new = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
        
        #輪郭検出
        det_img, contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), LWEIGHT) 
        cv2.imshow("gray", scale_down(im, max(img.shape[0], img.shape[1])/SIZE))
        cv2.waitKey(1)

        for cnt in contours:
            M = cv2.moments(cnt)
            areas.append(M['m00'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                stars.append(np.array([[cx, cy]], dtype='int32'))
            else:
                stars.append(np.array([cnt[0][0]], dtype='int32'))

        if len(stars) > THREASH*adapt and thr <= 250:
            #print("len:",len(stars))
            thr += 10
            del new
            #print("threshold:",thr)
        elif len(stars) == 0: #thr=260になるとここにくる
            thr = 250
            #ガンマを強くしつつ、星の数の制限を緩める
            gam *= 0.8
            adapt *= 1.2
            #print("try gamma:", gam)
            del new, img_gray
            dark = darken(image, gam)
            img_gray = cv2.cvtColor(dark, cv2.COLOR_RGB2GRAY)
        else:
            #print("len:",len(stars))
            flag = False
    #星のうち明るいほうから順に100取り出す
    r_areas_arg = np.argsort(areas)[::-1]
    for i in range(STARSIZE):
        astars.append(stars[r_areas_arg[i]])
    #print("threashold:",thr)
    #print("len:",len(astars))
    return astars

def on_mouse(event, x, y, flag, param):
    """マウスクリック時"""
    #左クリックで最近傍の星出力
    if event == cv2.EVENT_LBUTTONDOWN:
        print("mouse:", x, y, sep=' ', end='\n')
        print(search_near_star(x, y, 0, param))

def search_near_star(x, y, i, stars):
    """(x, y)にi番目(0オリジン)に近いものを返す"""
    if i >= len(stars):
        print("Can't detect")
        #sys.exit(1)
        return np.array([None, None])

    p = np.array([x, y])
    L = np.array([])
    for star in stars:
        L = np.append(L, np.linalg.norm(star-p))
    index = np.array(L)
    index = np.argsort(index)
    return stars[index[i]]

def draw_line(img, stars, constellation):
    C = constellation
    global likelihood
    global star_count
    navg = 0
    med = []
    ma = []
    for star in stars:
        s = np.array([star[0][0], star[0][1]])
        ns = search_near_star(s[0], s[1], 1, stars)
        ms = search_near_star(s[0], s[1], int(DEPTH/2), stars)
        ma.append(np.linalg.norm(ms - s))
        med.append(np.linalg.norm(ns - s))
    #navg /= len(stars)
    lowmed = np.median(med)
    midmax = np.amax(ma)

    stella_count = 0
    stella_data, like_list = [], []
    for star in stars:
        star_count = 1
        std = np.array([star[0][0], star[0][1]])
        i = 1
        while True:
            #2番目の星候補
            p1 = search_near_star(std[0], std[1], i, stars)[0]
            d1 = np.linalg.norm(std-p1)
            """
            if d1 < lowmed:
                i += 1
                continue
            elif i > DEPTH or d1 > midmax:
                break #次の根(基準星)へ
            """
            if i > DEPTH:
                break
            #2番目の星から先で星座が書けるかどうかをチェック
            point, bector = p1, p1-std
            #print(p1)
            likelihood, star_count = 0, 0
            point, bector = trac_constellation(False, img, point, bector, std, d1, stars, C)
            if star_count > 0:
                l_c = likelihood/star_count
                #print("L:",likelihood,"C:",star_count,"L/C:",l_c)
            C["itr"] = 0
            #第一返値で見つかってるかチェック
            if point is None: #見つかってなければ次の星へ
                i += 1
            else: #見つかったら
                if l_c < 1:
                    #描く
                    sp, ep = line_adjust(std, p1)
                    cv2.line(img, sp, ep, WHITE, LWEIGHT, cv2.LINE_AA)
                    cv2.circle(img, (std[0],std[1]), CRADIUS, WHITE, LWEIGHT, cv2.LINE_AA)
                    trac_constellation(True, img, p1, p1-std, std, d1, stars, C)
                    return
                elif l_c < 2 or star_count > C["N"]:
                    stella_count += 1
                    stella_data.append([p1, p1-std, std, d1])
                    like_list.append(l_c)
                i += 1
    print("visited all stars")
    if len(like_list) > 0:
        I = stella_data[np.argmin(like_list)]
        sp, ep = line_adjust(I[2], I[0])
        cv2.line(img, sp, ep, WHITE, LWEIGHT, cv2.LINE_AA)
        cv2.circle(img, (I[2][0],I[2][1]), CRADIUS, WHITE, LWEIGHT, cv2.LINE_AA)
        trac_constellation(True, img, I[0], I[1], I[2], I[3], stars, C)
    else:
        print("failed to detect")

def line_adjust(start, end):
    """線分を円周の部分までで止めるような始点、終点を返す"""
    b = end - start
    b = b / np.linalg.norm(b)
    restart = start + b * CRADIUS

    b = start - end
    b = b / np.linalg.norm(b)
    reend = end + b * CRADIUS

    return ((int(restart[0]), int(restart[1])), (int(reend[0]), int(reend[1])))

def trac_constellation(write, img, bp, bec, std_p, std_d, stars, constellation):
    """(描画判断、描画先、前の座標、前ベクトル、基準点、基準距離、星座標リスト、星座dic)"""
    C = constellation
    dist, ang, rd = C["D"][C["itr"]], C["ANGS"][C["itr"]], C["STD_D"][C["itr"]]

    i, p, d = 1, 0, 0
    angles, lengths = [], []
    A = []
    points = []
    global likelihood
    global star_count
    while d/std_d < dist * 0.9:
        p = search_near_star(bp[0], bp[1], i, stars)[0]
        if p is None:
            break
        else:
            d = np.linalg.norm(bp - p)
            i += 1
    while d/std_d < dist * 1.1:
        if p is None:
            break
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
            # TODO:角度の許容範囲
            if ((theta > ang-ARANGE and theta < ang+ARANGE) and
                (d_s > rd*0.9 and d_s < rd*1.1)): 
                A.append(theta)
                angles.append(abs(theta-ang))
                lengths.append(abs(d_s-rd))
                points.append([p[0], p[1]])
                
                #print(bp, i, "in", p,"(theta, d_s)", (theta, d_s),d/std_d, sep=" ")
                p = search_near_star(bp[0], bp[1], i, stars)[0]
                if p is None:
                    # TODO:応急
                    print("miss")
                    return (None, None)
                d = np.linalg.norm(bp - p)
                i += 1
            else:
                #print(bp, i, "out", p, "(theta, d_s)", (theta, d_s), sep=" ")
                p = search_near_star(bp[0], bp[1], i, stars)[0]
                if p is None:
                    # TODO:応急
                    print("miss")
                    return (None, None)
                d = np.linalg.norm(bp - p)
                i += 1
    if len(angles) == 0:
        #print("itr:", C["itr"], "angles is empty", ang)
        C["itr"] = 0
        C["BP"].clear()
        if write:
            # TODO:理想値を計算し線のみ描画
            cv2.circle(img, (bp[0], bp[1]), CRADIUS, WHITE, LWEIGHT, cv2.LINE_AA)
        # TODO:要検証 失敗時にも分岐を書く
        #if write and (len(C["BP"]) > 0):
        if len(C["BP"]) > 0:
            for (branch, rest) in zip(C["BP"], C["REST"]):
                trac_constellation(write, img, branch, tp-bp, std_p, std_d, stars, rest)    
            C["itr"] = 0
        return (None, None)
    else: #可能性のある星を検出できていた場合
        tp = np.array(points[np.argmin(angles)])
        star_count += 1
        if star_count <= 4:
            likelihood += (abs(d/std_d - dist) + np.min(angles) + lengths[np.argmin(angles)])/(star_count/C["N"])
        if write:
            #print("writed:", tp)
            sp, ep = line_adjust(bp, tp)
            cv2.line(img, sp, ep, WHITE, LWEIGHT, cv2.LINE_AA)
            cv2.circle(img, (bp[0], bp[1]), CRADIUS, WHITE, LWEIGHT, cv2.LINE_AA)

        if C["itr"] in C["JCT"]:
            C["BP"].append(tp)

        C["itr"] += 1
        if C["itr"] == len(C["D"]):
            cv2.circle(img, (tp[0], tp[1]), CRADIUS, WHITE, LWEIGHT, cv2.LINE_AA)
            #検出部終了時描画モードかつ分岐点が存在したら続きを描画
            if len(C["BP"]) > 0:
                for (branch, rest) in zip(C["BP"], C["REST"]):
                    #print("nowonBP:", branch)
                    trac_constellation(write, img, branch, tp-bp, std_p, std_d, stars, rest)    
            C["itr"] = 0
            #print("end checked")
            return (END, END)

        return trac_constellation(write, img, tp, tp-bp, std_p, std_d, stars, C)

if __name__ == '__main__':
    start = time.time()
    IMAGE_FILE = "1499" #スピード:test < 1618 <= 1614 << 1916
    f = "source\\" + IMAGE_FILE + ".JPG"
    img = cv2.imread(f)
    cs = Constellation.Sagittarius()
    #cs = Constellation.Perseus()
    
    #BIGMODE
    stars = detect_stars(img)
    draw_line(img, stars, cs.get())
    img = scale_down(img, max(img.shape[0], img.shape[1])/SIZE)
    """
    
    #SMALLMODE
    img = scale_down(img, max(img.shape[0], img.shape[1])/SIZE)
    stars = detect_stars(img)
    draw_line(img, stars, cs.get())
    """
    #cv2.namedWindow("stardust", cv2.WINDOW_NORMAL)
    cv2.imshow("stardust", img)
    cv2.setMouseCallback("stardust", on_mouse, stars)
    print("time:", time.time()-start)
    cv2.imwrite(cs.get_name() + "_" + IMAGE_FILE + ".JPG", img)
    cv2.waitKey()