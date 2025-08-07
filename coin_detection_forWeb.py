import cv2
import numpy as np
import statistics
import os
import streamlit as st
import tempfile

def detect_circles(img, img_output):
    # ガウシアンフィルタによる平滑化でノイズ除去
    img_blur = cv2.GaussianBlur(img, (7, 7), None)
    
    # Canny法によるエッジ検出
    img_edge = cv2.Canny(img_blur, 100, 300)
    
    # Hough変換による円検出
    circles = cv2.HoughCircles(
        img_edge,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,
        param2=20,
        minRadius=30,
        maxRadius=100
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # 表示用の画像に検出した円を描画
        for circle in circles[0, :]:
            # 円周を描画
            cv2.circle(img_output, (circle[0], circle[1]), circle[2], (0, 165, 255), 5)
            
            # 中心点を描画
            cv2.circle(img_output, (circle[0], circle[1]), 2, (0, 0, 255), 3)       
    else:
        print("No circles found")
    return circles, img_output


def detect_holes(img, circles, img_output):
    # 各円の穴情報を保持するリストを初期化
    holes_list = []
    
    # 各円に対して穴検出
    for circle in circles[0, :]:
        # 円の内側をマスク
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.circle(mask, (circle[0], circle[1]), circle[2]-5, 255, -1)
        roi = cv2.bitwise_and(img, img, mask=mask)

        # 円が含まれる最小矩形を切り出し
        x1 = max(circle[0] - circle[2], 0)
        y1 = max(circle[1] - circle[2], 0)
        x2 = min(circle[0] + circle[2], img_gray.shape[1])
        y2 = min(circle[1] + circle[2], img_gray.shape[0])
        roi_crop = roi[y1:y2, x1:x2]

        # ガウシアンフィルタによる平滑化でノイズ除去
        roi_blur = cv2.GaussianBlur(roi_crop, (9, 9), None)

        # Canny法によるエッジ検出
        roi_edge = cv2.Canny(roi_blur, 100, 300)

        # Hough変換による円検出
        holes = cv2.HoughCircles(
            roi_edge,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=100,
            param2=10,
            minRadius=5,
            maxRadius=circle[2]//2
        )

        hole = None
        
        if holes is not None:
            holes = np.uint16(np.around(holes))
            for idx, ic in enumerate(holes[0, :]):
                
                # 検出したの穴の中心と外円の中心とのユークリッド距離を計算
                x1f, icxf, cxf = float(x1), float(ic[0]), float(circle[0])
                y1f, icyf, cyf = float(y1), float(ic[1]), float(circle[1])
                dist = np.sqrt((x1f + icxf - cxf)**2 + (y1f + icyf - cyf)**2)
                # dist = np.sqrt((x1 + ic[0] - circle[0])**2 + (y1 + ic[1] - circle[1])**2)

                if ic[2] / circle[2] > 0.17 and ic[2] / circle[2] < 0.28 and dist < (circle[2] * 0.3):
                    # 穴の円周を描画
                    cx, cy, cr = x1 + ic[0], y1 + ic[1], ic[2]
                    cv2.circle(img_output, (cx, cy), cr, (0, 255, 0), 2)
                    
                    hole = holes[0, idx]

        holes_list.append(hole)

    return holes_list, img_output


def get_hsv_from_coins(img_bgr, circles, holes_list):
    # HSVに変換
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 各画素のhsvを保持するリストを初期化
    coin_hsv_list = []
    
    for idx, circle in enumerate(circles[0, :]):
        # マスク画像の初期化
        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        
        # 円の内側をマスク
        cv2.circle(mask, (circle[0], circle[1]), circle[2]-5, 255, -1)

        # 穴の内側を0にする
        if holes_list[idx] is not None:
            ic = holes_list[idx]
            cx = ic[0] + max(circle[0] - circle[2], 0)
            cy = ic[1] + max(circle[1] - circle[2], 0)
            cr = ic[2]
            cv2.circle(mask, (cx, cy), cr, 0, -1)
    
        # マスクされたコイン部分のHSV画素値リストを取得
        coin_pixels = img_hsv[mask == 255]
        coin_hsv_list.append(coin_pixels)
        
    return coin_hsv_list


def identify_coin_types(circles, holes_list, coin_hsv_list, img_output):
    # 各コインの種類を格納するリストを初期化
    coin_types = []
    
    # 各コインごとに，穴の有無と，色相・彩度によって種類を判別
    for idx, coin_pixels in enumerate(coin_hsv_list):
        # コイン画素の色相
        h_values = coin_pixels[:, 0]
        
        # コイン画素の彩度
        s_values = coin_pixels[:, 1]
        
        # 色相の最頻値
        h_mode = statistics.mode(h_values)
        
        # 彩度の最頻値
        s_mode = statistics.mode(s_values)
        
        # コインの外円情報(描画に使用)
        circle = circles[0, idx]

        # 推測された種類の初期化
        guess_type = 0
        
        # 穴の有無
        if holes_list[idx] is not None:
            # 彩度によって 5円 or 50円 を判別
            if s_mode > 100:
                guess_type = 5
            else:
                guess_type = 50
        else:
            # 色相によっって (500円, 100円) or 1円 or 10円　を判別
            if h_mode > 18 and h_mode < 34:
                # 彩度によって 500円 or 100円 を判別
                if s_mode > 50:
                    guess_type = 500
                else:
                    guess_type = 100
                    
            elif h_mode >= 34:
                guess_type = 1
                       
            else:
                guess_type = 10

        # 推測された種類を判別結果として格納
        coin_types.append(guess_type)
        
        # 判別結果を描画
        cv2.putText(img_output, str(coin_types[idx]), (circle[0], circle[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # 合計金額を描画
    cv2.putText(img_output, 'total: ' + str(sum(coin_types)) + ' yen', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return coin_types, img_output


if __name__ == '__main__':
    st.title('Yen Coins Detector')
    # streamlitからの画像のアップロード
    filepath_input = st.file_uploader('coins_image', type=['png', 'jpeg', 'jpg'])
    st.header('Result')
    if filepath_input:
        # バイト列として読み込み
        image_bytes = filepath_input.read()
    
        # 画像読み込み
        img_src = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR_BGR)
    
        # 画像サイズ変更
        img_resized = cv2.resize(img_src, (500, 500))
        
        # 表示用に画像をコピー
        img_dst = img_resized.copy()
        
        # グレースケールに変換
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 硬貨の円を検出
        circles, img_dst = detect_circles(img_gray, img_dst)

        # 硬貨の穴を検出
        holes, img_dst = detect_holes(img_gray, circles, img_dst)
        
        # 硬貨のHSVを取得
        coin_hsv = get_hsv_from_coins(img_resized, circles, holes)
        
        # 各硬貨の種類を判定
        coin_types, img_dst = identify_coin_types(circles, holes, coin_hsv, img_dst)
        
        st.image(img_dst, channels="BGR")