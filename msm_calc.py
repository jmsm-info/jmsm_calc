import discord
import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import pyocr
import re
import datetime
import textwrap
import configparser

from unicodedata import east_asian_width
wlen = lambda s: len([c for c in s if east_asian_width(c) in 'FWA'])
ljust = lambda s, width, *args: s.ljust(width - wlen(s), *args)
rjust = lambda s, width, *args: s.rjust(width - wlen(s), *args)

base_path = os.path.dirname(os.path.abspath(__file__)) + '/'
logs = 'logs/log.txt'

config_ini = configparser.ConfigParser()
config_ini.read(f'{base_path}/config.ini', encoding='utf-8')
TOKEN = config_ini["default"]["token"]
print(TOKEN)

client = discord.Client()

def logger(s):
    _s = f'{datetime.datetime.now()}, {s}'
    with open(base_path + logs, mode='a', encoding='UTF-8') as f:
        f.write(_s + '\n')
    try:
        print(_s)
    except UnicodeEncodeError:
        print(_s.encode('cp932','replace'))

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

# urlの画像をダウンロードしてPILで返す
#def dl2pl(url):
#    r = requests.get(url, stream=True)
#    if r.status_code == 200:
#        logger(r.url)
#        img = Image.open(BytesIO(r.content))
#        return img
#
#    return False

# urlの画像をダウンロードしてPILで返す
def dl2pl(url):
    filetype = ['image/png', 'image/jpeg']
    r = requests.head(url, stream=True)
    if r.status_code == 200:
        if 10000000 < int(r.headers['Content-Length']):
            err_msg = 'ファイルサイズがでかすぎます。10MB以下でお願いします'
            logger('[ERROR] ファイルサイズでかすぎ')
            return err_msg, False
        
        for i in filetype:
            if r.headers['Content-Type'] == i:
                break
        else:
            err_msg = f'対応するファイルが添付されていません\n 画像を添付したのにこのメッセージが出る場合は https://twitter.com/jmsm_info にお問い合わせください\n'
            logger('[ERROR] 未対応ファイルの添付')
            return err_msg, False
        
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            logger(r.url)
            img = Image.open(BytesIO(r.content))
            return '', img
    return '', False

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

#    print(np.count_nonzero(img_crop == img_find)/2520)
    if np.count_nonzero(img_crop == img_find)/2520 <= 0.89:
        return False

#    cv2.imwrite(f'{base_path}/img/tmp.png', img_crop)
    return True

# ダメージを計算
def calc_dmg(aab,phy = True, skill_base_dmg = 0, skill_dmg = 100):
    if phy:
        p = 0
    else:
        p = 3
#    dmg = (skill_base_dmg + aab[p+0] * (skill_dmg * (100 + aab[6]) + aab[p+1])) * (100 + aab[p+2]) * (100 + aab[11])
#    return int(dmg/100000000), int(dmg*(120 + aab[9])/10000000000)
#    dmg = (skill_base_dmg + aab[p+0] * (skill_dmg/100 * (1 + aab[6]/100) + aab[p+1]/100)) * (1 + aab[p+2]/100) * (1 + aab[11]/100)
#    return int(dmg), int(dmg*(1.2 + aab[9]/100))
    dmg = (skill_base_dmg + aab[p+0] * (skill_dmg/100 * (1 + aab[6]/100) + aab[p+1]/100)) * (1 + aab[p+2]/100) * (1 + aab[10]/100)
    return int(dmg), int(dmg*(1.2 + aab[8]/100))

# 戦闘力の増減分を再計算
def calc_combat(aab, _aab, combat_power, phy = True):
    if phy:
        p = 0
    else:
        p = 3
    # 元の攻撃力の増加分を引いて、HPと防御力の増加分だけにする
#    combat_power -= aab[p+0]*(100+aab[p+1]+1.5*(aab[p+2]+aab[6]+aab[8])+2*aab[9]+3*aab[11])/100
    combat_power -= aab[p+0]*(100+aab[p+1]+1.5*(aab[p+2]+aab[6]+aab[7])+2*aab[8]+3*aab[10])/100

    # 新しい攻撃力の増加分を計算し、足す
#    combat_power += _aab[p+0]*(100+_aab[p+1]+1.5*(_aab[p+2]+_aab[6]+_aab[8])+2*_aab[9]+3*_aab[11])/100
    combat_power += _aab[p+0]*(100+_aab[p+1]+1.5*(_aab[p+2]+_aab[6]+_aab[7])+2*_aab[8]+3*_aab[10])/100    

    return int(combat_power)
#旧
# 0 物理攻撃力
# 1 物理攻撃力上昇
# 2 物理ダメージ上昇
# 3 魔法攻撃力
# 4 魔法攻撃力上昇
# 5 魔法ダメージ上昇
# 6 対ボス攻撃力上昇
# 7 対プレイヤー攻撃力上昇
# 8 クリティカル率
# 9 クリティカルダメージ
#10 最大ダメージ
#11 最終ダメージ

#新
# 0 物理攻撃力
# 1 物理攻撃力上昇
# 2 物理ダメージ上昇
# 3 魔法攻撃力
# 4 魔法攻撃力上昇
# 5 魔法ダメージ上昇
# 6 対ボス攻撃力上昇
# 7 クリティカル率
# 8 クリティカルダメージ
# 9 最大ダメージ
#10 最終ダメージ
#11 防御力無視率


#unit = ['','%','%','','%','%','%','%','%','%','','%']
#aab_desc = ['物理攻撃力','物理攻撃力上昇','物理ダメージ上昇','魔法攻撃力','魔法攻撃力上昇','魔法ダメージ上昇','対ボス攻撃力上昇','対プレイヤー攻撃力上昇','クリティカル率','クリティカルダメージ','最大ダメージ','最終ダメージ']
unit = ['','%','%','','%','%','%','%','%','','%','%']
aab_desc = ['物理攻撃力','物理攻撃力上昇','物理ダメージ上昇','魔法攻撃力','魔法攻撃力上昇','魔法ダメージ上昇','対ボス攻撃力上昇','クリティカル率','クリティカルダメージ','最大ダメージ','最終ダメージ','防御力無視率']

def res_text(aab,combat_power, skill_base_dmg = 0, skill_dmg = 100):
    _s = ''
    for i,s in enumerate(unit):
        if s == '%':
            _s += '{0} {1: >9}%\n'.format(ljust(aab_desc[i],24), aab[i])
        else:
            _s += '{0: <12} {1: >10,}\n'.format(ljust(aab_desc[i],24), int(aab[i]))

    _s += '\n'
#    _s += '{0} {1: >10,}\n'.format(ljust('⚔戦闘力',24),combat_power)
#
#    d,_d = calc_dmg(aab)
#    _s += '🗡物理\n'
#    _s += '{0} {1: >10,}\n'.format(ljust('通常ダメージ',24), int(d * 0.95))
#    _s += '{0: <22} ～{1: >10,}\n'.format(' ', int(d * 1.049))
#    _s += '{0} {1: >10,}\n'.format(ljust('クリティカルダメージ',24), int(_d * 0.95))
#    _s += '{0: <22} ～{1: >10,}\n'.format(' ', int(_d * 1.049))
#    d,_d = calc_dmg(aab, phy=False)
#    _s += '🧙魔法\n'
#    _s += '{0} {1: >10,}\n'.format(ljust('通常ダメージ',24), int(d * 0.95))
#    _s += '{0: <22} ～{1: >10,}\n'.format(' ', int(d * 1.049))
#    _s += '{0} {1: >10,}\n'.format(ljust('クリティカルダメージ',24), int(_d * 0.95))
#    _s += '{0: <22} ～{1: >10,}\n'.format(' ', int(_d * 1.049))

    _s += '⚔戦闘力\n'
    _s += '{0: >10,}\n'.format(combat_power)
    _s += '\n'
    _s += '{0} {1}% で計算\n'.format('スキルダメージ', skill_dmg)
    _s += '{0} {1:,} で計算\n'.format('スキルベースダメージ', int(skill_base_dmg))

    _aab = aab[:]
    _aab[6] = 0

    _s += '🗡物理\n'
    d,_d = calc_dmg(_aab, skill_base_dmg = skill_base_dmg, skill_dmg = skill_dmg)
    _s += '通常,通常ダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(d * 0.95), int(d * 1.049))
    _s += '通常,クリティカルダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(_d * 0.95), int(_d * 1.049))
    d,_d = calc_dmg(aab, skill_base_dmg = skill_base_dmg, skill_dmg = skill_dmg)
    _s += '対ボス,通常ダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(d * 0.95), int(d * 1.049))
    _s += '対ボス,クリティカルダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(_d * 0.95), int(_d * 1.049))
    _s += '\n'
    _s += '🧙魔法\n'
    d,_d = calc_dmg(_aab, skill_base_dmg = skill_base_dmg, skill_dmg = skill_dmg, phy=False)
    _s += '通常,通常ダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(d * 0.95), int(d * 1.049))
    _s += '通常,クリティカルダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(_d * 0.95), int(_d * 1.049))
    d,_d = calc_dmg(aab, skill_base_dmg = skill_base_dmg, skill_dmg = skill_dmg, phy=False)
    _s += '対ボス,通常ダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(d * 0.95), int(d * 1.049))
    _s += '対ボス,クリティカルダメージ\n'
    _s += '{0: >10,}　～ {1: >10,}\n'.format(int(_d * 0.95), int(_d * 1.049))

    return _s

async def res(message):
    if message.attachments:
        for attachment in message.attachments:
            if attachment.url.endswith(("png", "jpg", "jpeg")):
                url_img = attachment.url
                break
        else:
            msg = await message.channel.send(f'対応するファイルが添付されていません\n 画像を添付したのにこのメッセージが出る場合は https://twitter.com/jmsm_info にお問い合わせください\n') 
            logger('[ERROR] 未対応ファイルの添付')
            return

    else:
        m = re.search(r'http(s?)://[0-9a-zA-Z?=#+_&:/.%-]+', message.content)
        if m is None:
            s = f'ファイルが添付されていません\nダメージ計算式は https://kaedemarket.web.fc2.com/JewelCompare/JewelCompare.html からの引用です\n問い合わせは https://twitter.com/jmsm_info まで\n'
            s += textwrap.dedent("""\
                > ---使い方---
                > 攻撃タイプ(物理攻撃力～最終ダメージ上昇)のキャラクター能力値のスクリーンショットを添付もしくはそのURLを貼り付けることで数値を読み取り、ダメージの計算を行います
                > よくあるミスとして、スクロールしていないスクリーンショットを送ったために最終ダメージ上昇のコンマが読み取れず数値が10倍になっている例があるので注意してください
                > また、ダメージ計算を実数値に近づけるためには使用するスキルのスキルダメージを記入する必要があります

                > ---使用例---
                ```
                https://... (画像のURL, 画像ファイルを添付した場合は不要)
                物理攻撃力 90000
                クリティカルダメージ +30%
                最終ダメージ上昇 -10%
                スキルダメージ 300%

                魔法ダメージ上昇 +10%	#メディテーション
                物理ダメージ上昇 +15%	#戻ってきて！
                魔法ダメージ上昇 +15%	#戻ってきて！
                物理ダメージ上昇 +15%	#コンバットオーダー
                魔法ダメージ上昇 +15%	#コンバットオーダー
                ```
                > スクリーンショットに書かれている数値を書き換えたい場合は上記のように指定します
                > 単位(%)は省くことが可能です
                > 符号なしの場合は絶対値として、符号あり(+ or -)の場合はスクリーンショットの値から相対的に計算します
                > 複数同じ能力名を符号ありで書いた場合はそれぞれ加算されます
                > ただし、スキルダメージとスキルベースダメージは絶対値として扱われます
                > また、各行の頭に書かれた能力名とその行にある数字以外は全て無視されるので使用例のようにコメントを書くことも可能です
                """)
            msg = await message.channel.send(s) 
            logger('[ERROR] ファイル添付なし')
            return
        else:
            url_img = m.group()

    err_msg, img = dl2pl(url_img)
    if img == False:
        if err_msg:
            msg = await message.channel.send(err_msg)
        else:
            msg = await message.channel.send(f'画像ファイルの取得に失敗しました\n 画像を添付したのにこのメッセージが出る場合は https://twitter.com/jmsm_info にお問い合わせください\n') 
            logger('[ERROR] 画像ファイルの取得に失敗')
        return


    img = img_pop(img)
    if 1280 <= img.width:
        img = scale_to_width(img,1280)
    else:
        pass
        msg = await message.channel.send(f'画像が小さすぎます\n 横幅1280px以上の画像のみ対応しています\n それ以上なのにこのメッセージが出る場合は https://twitter.com/jmsm_info にお問い合わせください\n') 
        logger('[ERROR] 画像が小さい')
        return


#    img.save(f'{base_path}/img/_tmp.png')

    if not check_screen_ability(img):
        msg = await message.channel.send('キャラクター能力値の画面を検出できませんでした\n キャラクター能力値の画面を添付したのにこのメッセージが出る場合は https://twitter.com/jmsm_info にお問い合わせください\n')
        logger('[ERROR] キャラクター能力値が検出できない')
        return 
    tools = pyocr.get_available_tools()
    tool = tools[0]
    builder = pyocr.builders.TextBuilder(tesseract_layout=6)

    # 戦闘力を取得
#    _img = img.crop((358,253,480,277))
    _img = img.crop((350,237,482,261))
    text = tool.image_to_string(_img, lang="eng", builder=builder)
    combat_power = int(re.sub(r"\D", "", text))

    # 各ステータスを取得
#    _img = img.crop((1100,196,1246,700))
    _img = img.crop((1092,193,1266,701))


    text = tool.image_to_string(_img, lang="eng", builder=builder)
    text = text.split()

    print(text)
    c = 0
    for i in range(0,len(text)-len(unit)+1):
        c = 0
        for j in range(i,len(unit)+i):
            if unit[c] == '%' and unit[c] == text[j][-1] or unit[c] != '%' and unit[c] != text[j][-1]:
                c += 1
            else:
                break
        if c == len(unit):
            break

    if c != len(unit):
        msg = await message.channel.send('画像を正しく検出できませんでした\n物理攻撃力～最終ダメージまで含んだ画像なのに、このメッセージが出る場合は https://twitter.com/jmsm_info にお問い合わせください\n')
        logger('[ERROR] 攻撃力のステータスが確認できない')
        return
    # 取ってきた値を格納
    _aab = text[i:i+c]
    aab = [0] * len(unit)
    for i,s in enumerate(unit):
        if s == '%':
            aab[i] = float(_aab[i][:-1])
        else:
            aab[i] = float(_aab[i].replace(',',''))

    # スキルダメージの指定があれば
#    try:
#        skill_dmg = float(message.content)
#    except ValueError:
#        skill_dmg = 100
#

    # ステータス変更の要望があれば
    _aab = aab[:]
    skill_dmg = 100
    skill_base_dmg = 0
#    for i,a in enumerate(aab_desc + ['スキルダメージ'] + ['スキルベースダメージ']):
#        for b in message.content.split('\n'):
#            if a == b[:len(a)]:
#                hit = re.search(r'[0-9]+(,|\.[0-9]|[0-9])*', b).group()
#                if a == 'スキルダメージ':
#                    skill_dmg = float(hit.replace(',',''))
#                elif a == 'スキルベースダメージ':
#                    skill_base_dmg = float(hit.replace(',',''))
#                else:
#                    _aab[i] = float(hit.replace(',',''))
#                break

    for b in message.content.split('\n'):
        for i,a in reversed(list(enumerate(aab_desc + ['スキルダメージ'] + ['スキルベースダメージ']))):
            if a == b[:len(a)]:
                hit = re.search(r'[-+]?[0-9]+(,|\.[0-9]|[0-9])*', b).group()
                if a == 'スキルダメージ':
                    skill_dmg = float(hit.replace(',',''))
                elif a == 'スキルベースダメージ':
                    skill_base_dmg = float(hit.replace(',',''))
                elif hit[0] == '+' or hit[0] == '-':
                    _aab[i] = float(_aab[i]) + float(hit.replace(',',''))
                else:
                    _aab[i] = float(hit.replace(',',''))
                break

#    print(res_text(_aab, calc_combat(aab, _aab, combat_power, phy = aab[0] > aab[3])))
    msg = await message.channel.send(f'```\n{res_text(_aab, calc_combat(aab, _aab, combat_power, phy = aab[0] > aab[3]), skill_dmg = skill_dmg, skill_base_dmg = skill_base_dmg)}```') 
    logger('[OK] 正常に完了')

@client.event
async def on_message(message):
    # メッセージ送信者がBotだった場合は無視する
    if message.author.bot:
        return
    if (type(message.channel) == discord.DMChannel) and (client.user == message.channel.me):
        logger(f'{message.author.id}, {message.author.name}#{message.author.discriminator}')
        await res(message)

@client.event
async def on_ready():
    # 起動したらターミナルにログイン通知が表示される
    print('ログインしました')
    await client.change_presence(activity=discord.Game(name="適当な文字を送ると使い方を返信します", type=1))

if __name__ == "__main__":
    pass
#    _img(Image.open(f'{base_path}/tmp/image3.jpg'))
    #msm_calc(img)

client.run(TOKEN)