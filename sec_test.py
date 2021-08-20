import os
from PIL import Image
import cv2
import numpy as np
import datetime
import glob

from unicodedata import east_asian_width
wlen = lambda s: len([c for c in s if east_asian_width(c) in 'FWA'])
ljust = lambda s, width, *args: s.ljust(width - wlen(s), *args)
rjust = lambda s, width, *args: s.rjust(width - wlen(s), *args)

base_path = os.path.dirname(os.path.abspath(__file__)) + '/'

def logger(s):
    _s = f'{datetime.datetime.now()}, {s}'
    print(_s)
#    with open(base_path + logs, mode='a') as f:
#        f.write(_s + '\n')

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def is_black(img, x1,y1,x2,y2):
    img_crop = img.crop((x1, y1, x2, y2))
    img = pil2cv(img_crop)
    if 0.99 <= np.count_nonzero(img == 0) / img.size:
        return True
    return False

# スケーリングの色が真っ黒じゃない場合があるので、真っ黒にする
def darker_than_black(img):
    x,y = img.size
    for i in range(x):
        for j in range(y):
            r,g,b = img.getpixel((i,j))
            if r < 10 and g < 10 and b < 10:
                img.putpixel((i,j), (0,0,0))
    return img

# 縦横の比率からpopupだけを切り抜く
def img_pop(img):
    std_aspect_ratio = 16/9
    src_aspect_ratio = img.width/img.height
    
    if src_aspect_ratio < std_aspect_ratio:
        #縦長(タブレット系)
#        print("タブレット系")
        p = (img.height - (img.width / std_aspect_ratio))/2
        img_crop = img.crop((0, p, img.width, img.height - p))
    
    elif src_aspect_ratio > std_aspect_ratio:
        #横長(スマホ系)
#        print("スマホ系")
#        p = (img.width - (img.height * std_aspect_ratio))/2
#        img_crop = img.crop((p, 0, img.width - p, img.height))
#        if 2 < src_aspect_ratio:
        if img.width == 2048 and img.height == 946:
            img_crop = img.crop((216, 0, 1832, 909))
        elif img.width == 2436 and img.height == 1125:
            img_crop = img.crop((258, 0, 2178, 1080))
        elif img.width == 2532 and img.height == 1170:
            img_crop = img.crop((266, 0, 2266, 1125))
        elif img.width == 1792 and img.height == 828:
            img_crop = img.crop((190, 0, 1602, 795))
        elif img.width == 2778 and img.height == 1284:
            img_crop = img.crop((293, 0, 2485, 1233))
        elif img.width == 2246 and img.height == 1080:
            _p = 88
            if is_black(img, 0, 0, 87, img.height):
                p = (img.width - _p - (img.height * std_aspect_ratio))/2
                img_crop = img.crop((p + _p, 0, img.width - p, img.height))
            elif is_black(img, 2158, 0, img.width, img.height):
                p = (img.width - _p - (img.height * std_aspect_ratio))/2
                img_crop = img.crop((p, 0, img.width - p - _p, img.height))
            else:
                p = (img.width - (img.height * std_aspect_ratio))/2
                img_crop = img.crop((p, 0, img.width - p, img.height))
        elif 2.163 <= src_aspect_ratio and src_aspect_ratio <= 2.165:
            #下にバーが出る系(iPhone)
#            print("一部iPhone")
            p = (img.width - (img.height * 0.958 * std_aspect_ratio))/2
            img_crop = img.crop((p, 0, img.width - p, img.height * 0.958))
        else:
            # アプリスケーリングしてそうなら少し画像を加工する
            if img.getbbox() != (0,0,img.size[0],img.size[1]):
                img = darker_than_black(img)
                img_range = img.getbbox()
                img = img.crop(img_range)
            p = (img.width - (img.height * std_aspect_ratio))/2
            img_crop = img.crop((p, 0, img.width - p, img.height))
    else:
        #16/9の端末
#        print("16/9の端末")
        img_crop = img
    return img_crop

# アスペクト比を固定して、幅が指定した値になるようリサイズする。
def scale_to_width(img, width):
    height = round(img.height * width / img.width)
    return img.resize((width, height))

def check_screen_ability(img):
    img = img.convert("RGB")

    # キャラクター能力値の画面か判定
    img_find = Image.open(f'{base_path}/img/キャラクター能力値.png')
    img_find = img_find.convert("RGB")

    img_crop = img.crop((22,26,50,56))

    img_find = pil2cv(img_find)
    img_crop = pil2cv(img_crop)

    ret, img_find = cv2.threshold(img_find, 128, 255, cv2.THRESH_BINARY)
    ret, img_crop = cv2.threshold(img_crop, 128, 255, cv2.THRESH_BINARY)

    print(np.count_nonzero(img_crop == img_find)/2520)
    if np.count_nonzero(img_crop == img_find)/2520 <= 0.89:
        return False

    cv2.imwrite(f'{base_path}/img/tmp.png', img_crop)
    return True

def res(img):
    img = img_pop(img)
    if 1280 <= img.width:
        img = scale_to_width(img,1280)
    else:
        pass
        logger('[ERROR] 画像が小さい')
        return

#    img.show()
    if not check_screen_ability(img):
        logger('[ERROR] キャラクター能力値が検出できない')
        img.show()
        return 

    print('OK')

if __name__ == "__main__":
#    res(Image.open(f'{base_path}/tmp/Screenshot_20210607-182210_M.jpg'))
#    res(Image.open(f'{base_path}/tmp/2436x1125.png'))

    files = glob.glob(f'{base_path}/tmp/*')
    for file in files:
        print(file.split('\\')[-1])
        res(Image.open(file))
