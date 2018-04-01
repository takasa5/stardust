"""stardust 2017/10/21"""
import numpy as np
import cv2
import time
import Constellation
IMPORT_SOCKET = True
try:
    from flask_socketio import emit
except ImportError:
    IMPORT_SOCKET = False
SIZE = 666 #画像サイズ(横)

class Stardust:
    def __init__(self, image_name,
                 star_num=120,
                 star_depth=5,
                 dist_max=50,
                 socket=None,
                 debug=False
                ):
        global IMPORT_SOCKET
        if isinstance(image_name, np.ndarray): # 画像が直接渡された場合
            self.image = image_name
        else:
            self.image = cv2.imread(image_name)
        # 小さすぎたら拡大
        if max(self.image.shape[0], self.image.shape[1]) < 1200:
            self.image = self.scale_down(self.image, max(self.image.shape[0], self.image.shape[1])/1200)
        self.star_num = star_num # Param:取り出す星の数
        self.star_depth = star_depth # Param:近隣探索数の上限
        self.dist_max = dist_max # Param:許容する予測誤差の上限
        self.likelihood = 0
        self.written_img = None
        self.stars_dist = {"now": np.array([-1, -1])}
        if IMPORT_SOCKET:
            self.socket = socket
        else:
            self.socket = None
        self.debug = debug
        self.stars = self.__detect_stars()
        
        
    def get_image(self):
        return self.written_img

    def scale_down(self, img, scale):
        """入力画像をscale分の1に縮小"""
        hight = img.shape[0]
        width = img.shape[1]
        small = cv2.resize(img, (round(width/scale), round(hight/scale)))
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
            # DEBUG
            if self.debug:
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
                if self.debug:
                    cv2.imshow("deleted", self.scale_down(del_img, max(del_img.shape[0], del_img.shape[1])/666))
                    cv2.waitKey(1)

                continue
            else:
                # DEBUG
                if self.debug:
                    cv2.imshow("deleted", self.scale_down(del_img, max(del_img.shape[0], del_img.shape[1])/666))
                    cv2.waitKey(1)
            #ここまで
            
            flag = False
            
        #星のうち明るいほうから順に取り出す
        r_areas_arg = np.argsort(areas)[::-1] #面積の大きい順にインデックスをリストに格納
        astars = [stars[r_areas_arg[i]] for i in range(self.star_num)]

        print("threashold:",thr)
        # DEBUG
        if self.debug:
            tmp = self.image.copy()
            for star in astars:
                cv2.circle(tmp, (star[0],star[1]), 2, (0,0,255), -1, cv2.LINE_AA)
            cv2.imshow("finalcnt", self.scale_down(tmp, max(tmp.shape[0], tmp.shape[1])/SIZE))
            cv2.waitKey(1)
        return astars

    def on_mouse(self, event, x, y, flag, param):
        """マウスクリック時(dup)"""
        #左クリックで最近傍の星出力
        if event == cv2.EVENT_LBUTTONDOWN:
            print("mouse:", x, y, sep=' ', end='\n')
            print(self.search_near_star((x, y), 0))

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

        min_like = 100
        best_point = None
        sockcnt = 0
        for star in self.stars:
            if self.socket is not None:
                emit('searching', {"data": sockcnt})
                sockcnt += 1
                self.socket.sleep(0)
            self.std_star = star
            i = 1
            while True:
                #2番目の星候補
                p1 = self.search_near_star(star, i)
                self.likelihood, self.star_count = 0, 1
                ret = self.__search_constellation(0, p1, p1 - self.std_star, constellation)
                if ret == constellation["MAX"] and self.likelihood / self.star_count < 5: # 全部見つかったら
                    # 1つめと2つめについて描く
                    print(self.star_count, self.likelihood / self.star_count)
                    p_list = [star, p1]
                    sp, ep = self.__line_adjust(star, p1)
                    cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
                    for p in p_list:
                        cv2.circle(self.written_img, 
                                   (p[0], p[1]),
                                   self.c_radius,
                                   (255, 255, 255),
                                   self.l_weight,
                                   cv2.LINE_AA    
                                  )          
                    self.__search_constellation(0, p1, p1 - self.std_star, constellation, write=True)
                    return
                elif ret > constellation["N"]:
                    print(self.star_count, self.likelihood / self.star_count)
                    if min_like > self.likelihood / self.star_count:
                        min_like = self.likelihood / self.star_count
                        best_point = [star, p1]

                i += 1
                if self.star_depth < i:
                    break
        
        if best_point is None:
            print("failed to detect")
        else:
            print(best_point, min_like)
            sp, ep = self.__line_adjust(best_point[0], best_point[1])
            cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
            for p in best_point:
                cv2.circle(self.written_img, 
                           (p[0], p[1]),
                           self.c_radius,
                           (255, 255, 255),
                           self.l_weight,
                           cv2.LINE_AA    
                          )          
            self.__search_constellation(0,
                                        best_point[1],
                                        best_point[1] - best_point[0],
                                        constellation,
                                        write=True
                                       )
        
    def __search_constellation(self, count, point, bector, constellation, write=False):
        """(何番目の星か, 前の点, 前のベクトル, 星座(の一部))"""
        dist, ang = constellation["D"][count], constellation["ANGS"][count]
        if count == 0: # TODO:応急処置 ちゃんと後始末ができるようにしたい
            constellation["BP"].clear()
        predict = point + self.__rotate_bector(bector, ang) * dist
        near_predict = self.search_near_star(predict, 0)
        predict_diff = np.linalg.norm(near_predict - predict)
        if self.star_count <= constellation["N"]:
            self.likelihood += predict_diff
        if count in constellation["JCT"]: # 現在の点が分岐点なら
            constellation["BP"].append(near_predict)

        if (predict_diff < self.dist_max
            and not np.allclose(near_predict, point)
            and not np.allclose(near_predict, self.std_star)
           ): # もし予想地点近く(近くとは)に星があれば
            self.star_count += 1
            if write:
                sp, ep = self.__line_adjust(point, near_predict)
                cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
                cv2.circle(self.written_img, 
                           (near_predict[0], near_predict[1]),
                           self.c_radius,
                           (255, 255, 255),
                           self.l_weight,
                           cv2.LINE_AA    
                          )

            if count+1 == len(constellation["D"]): # 端点ならば
                if len(constellation["BP"]) > 0: # 分岐点が存在すれば
                    for (branch, rest) in zip(constellation["BP"], constellation["REST"]):
                        self.__search_constellation(0, branch, near_predict - point, rest, write=write)

                return self.star_count
            return self.__search_constellation(count+1,
                                               near_predict,
                                               near_predict - point,
                                               constellation,
                                               write=write
                                              )
        else: # 近くに星がなければ
            constellation["BP"].clear()
            return self.star_count

    def __line_adjust(self, start, end):
        """線分を円周の部分までで止めるような始点、終点を返す"""
        b = end - start
        b = b / np.linalg.norm(b)
        restart = start + b * self.c_radius

        b = start - end
        b = b / np.linalg.norm(b)
        reend = end + b * self.c_radius

        return ((int(restart[0]), int(restart[1])), (int(reend[0]), int(reend[1])))

    def __rotate_bector(self, bector, deg):
        """bector を deg 度だけ回転する"""
        rad = np.deg2rad(deg)
        cos = np.cos(rad)
        sin = np.sin(rad)
        R = np.matrix((
            (cos, -sin),
            (sin, cos)
        ))
        return np.dot(R, bector)
 
if __name__ == '__main__':
    # TODO:g003 g004 にて問題 案：std_distが短いものは外したい
    IMAGE_FILE = "1916" #スピード:test < 1618 <= 1614 << 1916
    f = "source\\" + IMAGE_FILE + ".JPG"

    start = time.time()    
    sd = Stardust(f, debug=True)
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