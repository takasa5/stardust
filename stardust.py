"""stardust 2017/10/21~"""
import numpy as np
import cv2
import time
import Constellation as cs
IMPORT_SOCKET = True
try:
    from flask_socketio import emit
except ImportError:
    IMPORT_SOCKET = False
SIZE = 666 #画像サイズ(横)

class Stardust:
    def __init__(self, image_name,
                 star_num=120,
                 star_depth=10,
                 dist_max=50, # 画像の大きさによるので固定すべきでなさそう
                 angle_max=5,
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
        self.text_size = 1.72e-7 * self.image.shape[0] * self.image.shape[1] + 1.34 # TODO:adjust!
        self.text_weight = 3 if self.image.shape[0] > 2000 and self.image.shape[1] > 2000 else 1
        self.star_num = star_num # Param:取り出す星の数
        self.star_depth = star_depth # Param:近隣探索数の上限
        self.dist_max = dist_max # Param:許容する距離誤差の上限
        self.angle_max = angle_max # Param:許容する角度誤差の上限
        self.likelihood = 0
        self.written_img = self.image.copy()
        self.stars_dist = {"now": np.array([-1, -1])}
        if IMPORT_SOCKET:
            self.socket = socket
        else:
            self.socket = None
        self.debug = debug
        # 画像の4隅
        self.a = np.array([0, 0])
        self.b = np.array([self.image.shape[1] - 1, 0])
        self.c = np.array([self.image.shape[1] - 1, self.image.shape[0] - 1])
        self.d = np.array([0, self.image.shape[0] - 1])
        self.detect = None # 見つかったかどうか(探索後T/F)
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
            self.stars_dist["index"] = index #メモ化
            return self.stars[index[i]]

    def draw_line(self, constellation, mode=cs.DEFAULT, predict_circle=False, write_text=False):
        self.predict_circle = predict_circle
        if isinstance(constellation, list):
            for cst in constellation:
                self.draw_line(cst, mode=mode, predict_circle=predict_circle, write_text=write_text)
            return
        if mode == cs.DEFAULT:
            line = constellation.line
        elif mode == cs.IAU:
            line = constellation.iau
        self.constellation = constellation

        min_like = 0
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
                self.second_star = p1
                self.likelihood, self.star_count = 0, 1
                ret = self.__search_constellation(0, p1, p1 - self.std_star, line)
                correct_probability = self.likelihood / line["MAX"]
                if ret == line["MAX"] or correct_probability > 0.8: # 全部見つかったら
                    # 1つめと2つめについて描く
                    if self.debug:
                        print(self.std_star, self.star_count, round(correct_probability * 100, 2), "%" )
                    self.star_count = 0
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
                    self.__search_constellation(0, p1, p1 - self.std_star, line, write=True)
                    if write_text:
                        cv2.putText(self.written_img,
                                    constellation.en_name,
                                    (self.std_star[0] + 4 * self.c_radius, self.std_star[1] - 4 * self.c_radius),
                                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                                    self.text_size, (255,255,255), self.text_weight, cv2.LINE_AA # 太さをマネージする(星座の大きさの計算が必要…？)
                                   )
                    if self.socket is not None:
                        emit('searching', {"data": self.star_num-1})
                        self.socket.sleep(0)
                    print(constellation.en_name, "wrote.")
                    self.detect = True
                    return self.detect
                elif correct_probability > 0.5:
                    if self.debug:
                        print(self.std_star, self.star_count, round(correct_probability * 100, 2), "%" )
                    if min_like < correct_probability:
                        min_like = correct_probability
                        best_point = [star, p1]
                elif self.star_count > line["N"]:
                    print(self.std_star, self.star_count, round((self.likelihood / line["MAX"]) * 100, 2), "%" )

                i += 1
                if self.star_depth < i:
                    break
        self.star_count = 0
        if best_point is None:
            print("failed to detect", constellation.en_name)
            self.detect = False
            return self.detect
        else:
            self.std_star = best_point[0]
            self.second_star = best_point[1]
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
                                        line,
                                        write=True,
                                        predict_write=True
                                       )
            if write_text:
                cv2.putText(self.written_img,
                            constellation.en_name,
                            (self.std_star[0] + 4 * self.c_radius, self.std_star[1] - 4 * self.c_radius),
                            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                            self.text_size, (255,255,255), self.text_weight, cv2.LINE_AA # 太さをマネージする(星座の大きさの計算が必要…？)
                           )
            print(constellation.en_name, "wrote.")
            self.detect = True
            return self.detect

    def __search_constellation(self, count, point, bector, constellation, write=False, predict_write=False):
        """(何番目の星か, 前の点, 前のベクトル, 星座(の一部))"""
        dist, ang = constellation["D"][count], constellation["ANGS"][count]
        if count == 0: # TODO:応急処置 ちゃんと後始末ができるようにしたい
            constellation["BP"].clear()
        predict = point + self.__rotate_bector(bector, ang) * dist
        near_predict = self.search_near_star(predict, 0)
        predict_diff = np.linalg.norm(near_predict - predict)
        
        theta = self.__calc_angle(bector, near_predict - point)
        # もし予想地点近く(近くとは)に星があれば
        if (predict_diff < self.dist_max and abs(abs(ang) - theta) < self.angle_max):
            # 尤度の計算
            found_bec_rate = np.linalg.norm(near_predict - point) / np.linalg.norm(bector)
            err = abs(dist - found_bec_rate) if abs(dist - found_bec_rate) < 1 else 1
            self.likelihood += 1 - err / dist
            """
            if self.star_count - 1 <= constellation["N"]:
                tmplike = max(predict_diff / self.dist_max, abs(abs(ang) - theta) / self.angle_max)
                if tmplike < 0.2:
                    self.likelihood -= 0.2
                else:
                    self.likelihood += tmplike
            else:
                tmplike = max(predict_diff / self.dist_max, abs(abs(ang) - theta) / self.angle_max)
                if tmplike < 0.2:
                    self.likelihood -= 0.2
            """
            if count == 0 and (-2 in constellation["JCT"] or -1 in constellation["JCT"]): # 分岐点が基準点の時
                for i in range(constellation["JCT"].count(-2)): # TODO: -1のもなんとかしよう
                    constellation["BP"].append(self.std_star)
                for i in range(constellation["JCT"].count(-1)):
                    constellation["BP"].append(self.second_star) # なんとかしたつもり
            elif count in constellation["JCT"]: # 現在の点が分岐点なら
                for i in range(constellation["JCT"].count(count)):
                    constellation["BP"].append(near_predict)
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
                        self.__search_constellation(0,
                                                    branch,
                                                    near_predict - point,
                                                    rest,
                                                    write=write,
                                                    predict_write=predict_write
                                                   )

                return self.star_count
            return self.__search_constellation(count+1,
                                               near_predict,
                                               near_predict - point,
                                               constellation,
                                               write=write,
                                               predict_write=predict_write
                                              )
        elif (predict_write and write): # 予想描画機能オンのとき
            predict[0, 0] = int(predict[0, 0])
            predict[0, 1] = int(predict[0, 1])
            predict = np.array(predict.tolist())[0]
            # 予測点がはみ出すとき
            if ((predict[0] < 0 or predict[0] >= self.image.shape[1])
                or (predict[1] < 0 or predict[1] >= self.image.shape[0])
               ):
                # TODO: はみ出した点が分岐点だった場合の処理
                managed_predict = self.__manage_cross(point, predict)
                print("managed:", managed_predict)
                if managed_predict is not None:
                    sp, ep = self.__line_adjust(point, managed_predict)
                    cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
                if count+1 == len(constellation["D"]): # 端点ならば
                    if len(constellation["BP"]) > 0: # 分岐点が存在すれば
                        for (branch, rest) in zip(constellation["BP"], constellation["REST"]):
                            self.__search_constellation(0,
                                                        branch,
                                                        predict - point,
                                                        rest,
                                                        write=True,
                                                        predict_write=True
                                                    )

                    return self.star_count
                
                return self.__search_constellation(count+1,
                                                   predict,
                                                   predict - point,
                                                   constellation,
                                                   write=True,
                                                   predict_write=True
                                                  )
                
            if count == 0 and (-2 in constellation["JCT"]): # 分岐点が基準点の時
                for i in range(constellation["JCT"].count(-2)):
                    constellation["BP"].append(self.std_star)
                for i in range(constellation["JCT"].count(-1)):
                    constellation["BP"].append(self.second_star) # なんとかしたつもり
            elif count in constellation["JCT"]: # 現在の点が分岐点なら
                for i in range(constellation["JCT"].count(count)):
                    constellation["BP"].append(near_predict)
            #print(predict)
            sp, ep = self.__line_adjust(point, predict)
            cv2.line(self.written_img, sp, ep, (255,255,255), self.l_weight, cv2.LINE_AA)
            if self.predict_circle: # predict_circleによって予測時に円を描くか決める
                print("predict", predict)
                cv2.circle(self.written_img, 
                           (int(predict[0]), int(predict[1])),
                           self.c_radius,
                           (255, 255, 255),
                           self.l_weight,
                           cv2.LINE_AA    
                          )
            if count+1 == len(constellation["D"]): # 端点ならば TODO:端点いかなくても分岐点は書きたい→前の辺を参照する現状では厳しい
                if len(constellation["BP"]) > 0: # 分岐点が存在すれば
                    for (branch, rest) in zip(constellation["BP"], constellation["REST"]):
                        self.__search_constellation(0,
                                                    branch,
                                                    predict - point,
                                                    rest,
                                                    write=True,
                                                    predict_write=True
                                                   )

                return self.star_count
            return self.__search_constellation(count+1,
                                               predict,
                                               predict - point,
                                               constellation,
                                               write=True,
                                               predict_write=True
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
        return R @ bector
    
    def __calc_angle(self, bec_a, bec_b):
        """2つのベクトルのなす角を求める"""
        dot = bec_a @ bec_b
        cos = dot / (np.linalg.norm(bec_a) * np.linalg.norm(bec_b))
        rad = np.arccos(cos)
        return np.rad2deg(rad)

    def __check_cross(self, p1, p2, p3, p4):
        """ベクトル同士が交差していればTrue, else Falseを返す"""
        t1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
        t2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
        t3 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
        t4 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
        return t1 * t2 < 0 and t3 * t4 < 0

    def __manage_cross(self, start, end):
        """はみ出た点に対し、そこに続くような線を描くための枠上の座標を返す"""
        # TODO: 外から中へのクロスへの対処
        if self.__check_cross(start, end, self.a, self.b):
            print("0")
            a = start[1]
            deg = self.__calc_angle(end - start, self.b - self.a)
            if deg > 90:
                deg = 180 - deg
                x = a // np.tan(np.deg2rad(deg))
                return np.array([start[0] - x, 0])
            else:
                x = a // np.tan(np.deg2rad(deg))   
                return np.array(start[0] + x, 0)
        elif self.__check_cross(start, end, self.b, self.c):
            print("1")
            a = self.c[0] - start[0]
            deg = self.__calc_angle(end - start, self.c - self.b)
            if deg > 90:
                deg = 180 - deg
                x = a // np.tan(np.deg2rad(deg))
                return np.array([self.b[0], start[1] - x])
            else:
                x = a // np.tan(np.deg2rad(deg))
                return np.array([self.b[0], start[1] + x])
        elif self.__check_cross(start, end, self.c, self.d):
            print("2")
            a = self.d[1] - start[1]
            deg = self.__calc_angle(end - start, self.d - self.c)
            if deg > 90:
                deg = 180 - deg
                x = a // np.tan(np.deg2rad(deg))
                return np.array([start[0] - x, self.c[1]])
            else:
                x = a // np.tan(np.deg2rad(deg))
                return np.array([start[0] + x, self.c[1]])
        elif self.__check_cross(start, end, self.d, self.a):
            print("3")
            a = start[0]
            deg = self.__calc_angle(end - start, self.a - self.d)
            if deg > 90:
                deg = 180 - deg
                x = a // np.tan(np.deg2rad(deg))
                return np.array([0, start[1] + x])
            else:
                x = a // np.tan(np.deg2rad(deg))
                return np.array([0, start[1] - x])

if __name__ == '__main__':
    #test, 0004, 0038, 1499, 1618, 1614, 1916, g001 ~ g004, dzlm, dalr, daqw
    IMAGE_FILE = "g002"
    f = "source\\" + IMAGE_FILE + ".JPG"
    start = time.time()    
    sd = Stardust(f, debug=True)
    cst = cs.Sagittarius()
    #sd.draw_line(cst)
    sd.draw_line(cs.sco)
    end = time.time()
    print("elapsed:", end - start)
    ret = sd.get_image()
    cv2.namedWindow("return", cv2.WINDOW_NORMAL)
    cv2.imshow("return", ret)
    cv2.setMouseCallback("return", sd.on_mouse)
    #cv2.imwrite(cst.name+"_"+IMAGE_FILE+".JPG", ret)
    #cv2.imwrite("multi_"+IMAGE_FILE+".JPG", ret)
    cv2.waitKey()