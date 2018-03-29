"""stardust 2017/10/21"""
import numpy as np
import cv2
import math, os, sys, time
import Constellation

SIZE = 666 #画像サイズ(横)

class Stardust:
    def __init__(self, image_name,
                 star_num=120,
                 star_depth=5,
                 angle_depth=5,
                 likelihood_thr=2
                ):
        if isinstance(image_name, np.ndarray): # 画像が直接渡された場合
            self.image = image_name
        else:
            self.image = cv2.imread(image_name)
        # 小さすぎたら拡大
        if max(self.image.shape[0], self.image.shape[1]) < 1200:
            self.image = self.scale_down(self.image, max(self.image.shape[0], self.image.shape[1])/1200)
        self.star_num = star_num # Param:取り出す星の数
        self.star_depth = star_depth # Param:近隣探索数の上限
        self.angle_depth = angle_depth # Param:角度誤差の許容範囲(±)
        self.likelihood_thr = likelihood_thr # Param:尤度の許容値
        self.stars = self.__detect_stars()
        self.written_img = None
        self.stars_dist = {"now": np.array([-1, -1])}
        
    def get_image(self):
        return self.written_img

    def scale_down(self, img, scale):
        """入力画像をscale分の1に縮小"""
        hight = img.shape[0]
        width = img.shape[1]
        small = cv2.resize(img, (round(width/scale), round(hight/scale)))
        #self.image = small
        return small

    def darken(self, gamma):
        """ガンマ補正をかける　gammma < 1で暗くなる"""
        lut = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            lut[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
        image_gamma = cv2.LUT(self.image, lut)
        self.image = image_gamma

    def __detect_stars(self):
        """最適(？)スレッショルドを設定し、抽出した星座標のリストを返す"""
        flag = True
        thr = 250
        gam, adapt = 1, 1
        
        # 円の半径と線の太さをちょうどよくする
        self.c_radius = int(max(self.image.shape[0], self.image.shape[1])/250)
        self.l_weight = int(max(self.image.shape[0], self.image.shape[1])/1000)
        
        #輪郭検出用グレースケール画像生成
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        del_img = self.image.copy()
        firstflag = True
        while flag:
            stars, areas = [], []
            ret, new = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
            cv2.imshow("gray", self.scale_down(new, max(new.shape[0], new.shape[1])/666))
            cv2.waitKey(1)
            #輪郭検出
            det_img, contours, hierarchy = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            """
            im = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), self.l_weight) 
            cv2.imshow("contours", scale_down(im, max(img.shape[0], img.shape[1])/SIZE))
            #cv2.imshow("contours", im)
            cv2.waitKey(1)
            """
            #print(len(contours))
            # TODO: ここのマジックナンバーなんとかする
            if len(contours) < 400:
                thr -= 10
                continue
            #else:
                #return
            #各輪郭から重心および面積を算出
            for cnt in contours:
                M = cv2.moments(cnt)
                areas.append(M['m00'])

                #輪郭の重心を座標値としてリストに格納
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    stars.append(np.array([cx, cy], dtype='int32'))
                else:
                    stars.append(np.array(cnt[0, 0], dtype='int32'))
            maxarea_index = np.argmax(areas)
            # TODO:画像の大半を消去してしまうようならthrあげるべき/削除方式をやめるべき？
            """
            if areas[maxarea_index] > image.shape[0] * image.shape[1] / 6:
                thr += 10
                continue
            """
            # 偏差を求める
            if firstflag:
                area_std = np.std(areas)
                print("std:", area_std)
                #q75, q25 = np.percentile(areas, [75, 25])
                #iqr = q75 - q25
                firstflag = False
            # 面積の最大値周辺は見ない：ここから
            # 分散が大きい場合、外れ値を削除していく
            # TODO: 楕円で削除してみる
            if area_std > 100 and areas[maxarea_index] > 2.5 * area_std:
                cnt = contours[maxarea_index]
                x, y, w, h = cv2.boundingRect(cnt)
                #epsilon = 0.1 * cv2.arcLength(cnt, True)
                #approx = cv2.approxPolyDP(cnt, epsilon, True)
                del_img = cv2.rectangle(del_img, (x, y), (x+w, y+h), (255, 0, 0), -1)
                img_gray = cv2.cvtColor(del_img, cv2.COLOR_RGB2GRAY)
                # DEBUG
                cv2.imshow("deleted", self.scale_down(del_img, max(del_img.shape[0], del_img.shape[1])/666))
                cv2.waitKey(1)

                continue
            else:
                # DEBUG
                cv2.imshow("deleted", self.scale_down(del_img, max(del_img.shape[0], del_img.shape[1])/666))
                cv2.waitKey(1)
            #ここまで
            
            flag = False
            
        #星のうち明るいほうから順に取り出す
        r_areas_arg = np.argsort(areas)[::-1] #面積の大きい順にインデックスをリストに格納
        astars = [stars[r_areas_arg[i]] for i in range(self.star_num)]

        print("threashold:",thr)
        # DEBUG
        tmp = self.image.copy()
        for star in astars:
            cv2.circle(tmp, (star[0],star[1]), 2, (0,0,255), -1, cv2.LINE_AA)
        # DEBUG
        cv2.imshow("finalcnt", self.scale_down(tmp, max(tmp.shape[0], tmp.shape[1])/SIZE))
        cv2.waitKey(1)
        return astars

    def on_mouse(self, event, x, y, flag, param):
        """マウスクリック時"""
        #左クリックで最近傍の星出力
        if event == cv2.EVENT_LBUTTONDOWN:
            print("mouse:", x, y, sep=' ', end='\n')
            # print(self.search_near_star(x, y, 0))

    def search_near_star(self, p, i):
        """(x, y)にi番目(0オリジン)に近いものを返す"""
        if i >= len(self.stars):
            print("Can't detect")
            #return np.array([None, None])
            return None

        if np.allclose(self.stars_dist["now"], p):
            return self.stars[self.stars_dist["index"][i]]
        else:
            L = [np.linalg.norm(star-p) for star in self.stars]
            index = np.array(L)
            index = np.argsort(index)
            self.stars_dist["index"] = index   #メモ化
            return self.stars[index[i]]

    def draw_line(self, constellation):
        self.written_img = self.image.copy()
        self.constellation = constellation
        C = constellation

        stella_count = 0
        stella_data, like_list = [], []
        for star in self.stars:
            self.star_count = 1
            #std = np.array([star[0][0], star[0][1]])
            std = star
            i = 1
            while True:
                #2番目の星候補
                p1 = self.search_near_star(std, i)
                d1 = np.linalg.norm(p1-std)
                if i > self.star_depth:
                    break
                #2番目の星から先で星座が書けるかどうかをチェック
                point, bector = p1, p1-std
                #print(p1)
                self.likelihood, self.star_count = 0, 0
                point, bector = self.__trac_constellation(False, point, bector, std, d1, C)
                if self.star_count > 0:
                    l_c = self.likelihood/self.star_count
                    #print("L:",self.likelihood,"C:",self.star_count,"L/C:",l_c)
                C["itr"] = 0
                #第一返値で見つかってるかチェック
                if point is None: #見つかってなければ次の星へ
                    i += 1
                else: # 見つかったら
                    if l_c < self.likelihood_thr: # 一定以下の尤度なら即時描く
                        print(l_c)
                        sp, ep = self.__line_adjust(std, p1)
                        cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
                        cv2.circle(self.written_img,
                                   (std[0],std[1]),
                                   self.c_radius,
                                   (255,255,255),
                                   self.l_weight,
                                   cv2.LINE_AA
                                  )
                        self.__trac_constellation(True, p1, p1-std, std, d1, C)
                        return
                    # 「一定の基準は越えるが、それらしいもの」は保存しておく
                    elif l_c < self.likelihood_thr*2 or self.star_count > C["N"]:
                        print(l_c)
                        stella_count += 1
                        stella_data.append([p1, p1-std, std, d1])
                        like_list.append(l_c)
                    i += 1
        print("visited all stars")
        if len(like_list) > 0:
            I = stella_data[np.argmin(like_list)]
            print("likelihood:", like_list[np.argmin(like_list)])
            sp, ep = self.__line_adjust(I[2], I[0])
            cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
            cv2.circle(self.written_img,
                       (I[2][0],I[2][1]),
                       self.c_radius,
                       (255,255,255),
                       self.l_weight,
                       cv2.LINE_AA
                      )
            self.__trac_constellation(True, I[0], I[1], I[2], I[3], C)
        else:
            print("failed to detect")

    def __line_adjust(self, start, end):
        """線分を円周の部分までで止めるような始点、終点を返す"""
        b = end - start
        b = b / np.linalg.norm(b)
        restart = start + b * self.c_radius

        b = start - end
        b = b / np.linalg.norm(b)
        reend = end + b * self.c_radius

        return ((int(restart[0]), int(restart[1])), (int(reend[0]), int(reend[1])))

    def __trac_constellation(self, write, bp, bec, std_p, std_d, cst):
        """(描画判断、前の座標、前ベクトル、基準点、基準距離, 星座の一部)"""
        C = cst
        img = self.written_img
        dist, ang, rd = C["D"][C["itr"]], C["ANGS"][C["itr"]], C["STD_D"][C["itr"]]

        i, p, d = 0, 0, 0
        angle_diff = 400 # 適当な大きい値
        # 近傍の星との距離がtarget distance(dist)の9割を越えるまで辿る
        while d/std_d < dist * 0.9:
            i += 1
            p = self.search_near_star(bp, i)
            if p is None:
                break
            else:
                d = np.linalg.norm(bp - p)
                
        # 9割を越えたら、1.1割を越えるまで辿る
        while d/std_d < dist * 1.1:
            p = self.search_near_star(bp, i)
            if p is None:
                break
            d = np.linalg.norm(bp - p)
            i += 1
            dot = np.dot(bec, p-bp)
            cos = dot / (d * np.linalg.norm(bec))
            if cos > 1 or cos < -1:
                continue
            else:
                rad = np.arccos(cos)
                theta = np.rad2deg(rad)
                d_s = np.linalg.norm(p - std_p)/std_d
                # 角度が許容範囲であれば、後に使う値を取り出す
                if ((theta > ang-self.angle_depth and theta < ang+self.angle_depth) 
                    and (d_s > rd*0.9 and d_s < rd*1.1)): 
                    # TODO:append地獄、ここを改善することでスピードアップできそう
                    # 角度誤差が最小のものを記録しておく
                    tmp_diff = abs(theta - ang)
                    if tmp_diff < angle_diff:
                        angle_abs = theta
                        angle_diff = tmp_diff
                        len_diff = abs(d_s - rd)
                        best_point = p

        if angle_diff == 400: # 一個も次の星らしき星が見つからなかった場合
            #print("itr:", C["itr"], "angles is empty", ang)
            C["itr"] = 0
            C["BP"].clear()
            if write: # write=Trueのときにここに入るのは最後のみ(多分)なので円を描いておわる
                # TODO:理想値を計算し線のみ描画
                cv2.circle(img, (bp[0], bp[1]), self.c_radius, (255,255,255), self.l_weight, cv2.LINE_AA)
            # TODO:要検証 失敗時にも分岐を書く
            """
            if len(C["BP"]) > 0:
                for (branch, rest) in zip(C["BP"], C["REST"]):
                    self.__trac_constellation(write, branch, tp-bp, std_p, std_d, rest)    
                C["itr"] = 0
            """
            return (None, None)
        else: #可能性のある星を検出できていた場合
            tp = np.array(best_point)
            self.star_count += 1
            # TODO: 尤度計算式は熟考すること
            #if self.star_count <= 4:
            self.likelihood += (abs(d/std_d - dist)
                                + angle_diff
                                + len_diff
                                ) / (self.star_count/C["N"])
            if write:
                print("ANGS:", angle_abs)
                #print("writed:", tp)
                sp, ep = self.__line_adjust(bp, tp)
                cv2.line(img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
                cv2.circle(img, (bp[0], bp[1]), self.c_radius, (255,255,255), self.l_weight, cv2.LINE_AA)

            if C["itr"] in C["JCT"]:
                C["BP"].append(tp)

            C["itr"] += 1
            if C["itr"] == len(C["D"]):
                cv2.circle(img, (tp[0], tp[1]), self.c_radius, (255,255,255), self.l_weight, cv2.LINE_AA)
                #検出部終了時描画モードかつ分岐点が存在したら続きを描画
                if len(C["BP"]) > 0:
                    for (branch, rest) in zip(C["BP"], C["REST"]):
                        #print("nowonBP:", branch)
                        self.__trac_constellation(write, branch, tp-bp, std_p, std_d, rest)    
                C["itr"] = 0
                #print("end checked")
                return (2**16, 2**16)

            return self.__trac_constellation(write, tp, tp-bp, std_p, std_d, C)

if __name__ == '__main__':
    IMAGE_FILE = "g006" #スピード:test < 1618 <= 1614 << 1916
    f = "source\\" + IMAGE_FILE + ".JPG"

    start = time.time()    
    sd = Stardust(f)
    cs = Constellation.Sagittarius()
    sd.draw_line(cs.get())
    end = time.time()
    print("elapsed:", end - start)
    
    ret = sd.get_image()
    cv2.namedWindow("return", cv2.WINDOW_NORMAL)
    cv2.imshow("return", ret)
    #cv2.imwrite(cs.get_name()+"_"+IMAGE_FILE+".JPG", ret)
    cv2.setMouseCallback("return", sd.on_mouse)
    cv2.waitKey()