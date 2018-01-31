# 本脚本用于去除重复的图片文件（除文件名外，内容一致），image_path为原始图片存放路径，remove_path为被挑选出来的重复图片存放路径
# 一开始的思路仅仅是读入两张图片，计算图片的差值，速度慢到令人发指。。。
# 后来查资料才知道有hash这个东西，我果然做不了算法工程师，什么都不懂
# 改用hash之后，每张图片都有一个唯一的hash值，将该值放入set内，就可以进行图片去重了
# 后面进一步查资料，还发现有人实现了模糊hash，即两张大致一样的图片也能得到同一hash值，这样就能找到一些疑似相同的图片了
# by @TerryBryant in Jan. 31st, 2018


import os
import shutil
import hashlib
import argparse

def hash_remove_repeat(ori_image_path, dst_remove_path):
    clean_set = set()
    for filename in os.listdir(ori_image_path):
        image_path = os.path.join(ori_image_path, filename)
        image_ = open(image_path, "rb").read()
        hash_ = hashlib.md5(image_).hexdigest()

        if hash_ not in clean_set:
            clean_set.add(hash_)
        else:
            shutil.move(image_path, dst_remove_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='', help='')
    parser.add_argument('--remove_path', type=str, default='', help='')
    outname = parser.parse_known_args()
    image_path = outname[0].image_path
    remove_path = outname[0].remove_path

    hash_remove_repeat(image_path, remove_path)



