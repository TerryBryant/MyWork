# 遍历文件夹下所有的视频文件，保存截图
# 输入分别为源视频路径、保存截图的路径、截图的频率
# 输出每个视频的帧提取图，保存在以视频文件名命名的文件夹内
# 本文件是../OpenCV/snap_image_from_video.cpp的python版本

import os
import cv2
import argparse


def snap_image_from_video(sourceFile, dstFile, freq):
    for file in os.listdir(sourceFile):
        videoname = os.path.join(sourceFile, file)
        newfilename = file.split(".")[0]    # 视频文件的名字，后面用作图片所在路径的文件名，以及图片名的前缀

        # 新建文件夹的名称
        newdir = os.path.join(dstFile, newfilename)
        if not os.path.isdir(newdir):
            os.mkdir(newdir)

        # 读取视频文件，计算相关参数
        cap = cv2.VideoCapture(videoname)
        if not cap.isOpened():
            print("视频 %s 无法打开" % videoname)
            continue

        print("正在处理视频 %s..." % videoname)
        cnt = 0
        while(True):
            ret, frame = cap.read()
            if not ret:     # 读到了视频最后，退出循环
                break

            if cnt % freq == 0:
                savepath = os.path.join(newdir, "%s_%04d.jpg" % (newfilename, cnt / freq))
                cv2.imwrite(savepath, frame)
            cnt += 1


parser = argparse.ArgumentParser()
parser.add_argument('--sourceFile', type=str, default='', help='源视频路径')
parser.add_argument('--dstFile', type=str, default='', help='保存视频截图的路径')
parser.add_argument('--freq', type=str, default='', help='视频截图间隔')
par = parser.parse_known_args()

snap_image_from_video(par[0].sourceFile, par[0].dstFile, int(par[0].freq))
print("所有视频提取图片完成，结果保存在", par[0].dstFile)
