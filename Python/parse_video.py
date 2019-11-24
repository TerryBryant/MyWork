import cv2
import imagehash
from PIL import Image
import time



class Dura:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end


def calculate_image_similarity(img1, img2):
    phashvalue = imagehash.phash(img1) - imagehash.phash(img2)
    ahashvalue = imagehash.average_hash(img1) - imagehash.average_hash(img2)
    return ahashvalue


def get_hash_array(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Unable to open, ', video_path)
        return None

    hash_arr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (480, 270))
        frame = Image.fromarray(frame)

        hash_arr.append(imagehash.average_hash(frame))

    return hash_arr


def compare_hash_arr(arr1, arr2):
    cnt1 = 0
    cnt2 = 0
    # max_len = len(arr2)
    # if len(arr1) > max_len:
    #     max_len = len(arr1)

    # find common content
    com_arr1 = []
    com_arr2 = []
    d1 = Dura(cnt1, cnt1)
    d2 = Dura(cnt2, cnt2)

    while cnt1 < len(arr1) and cnt2 < len(arr2):
        diff = arr1[cnt1] - arr2[cnt2]

        if diff < 5:   # same image
            cnt1 += 1
            cnt2 += 1
        else:
            d1.end = cnt1
            d2.end = cnt2
            com_arr1.append((d1.begin, d1.end))
            com_arr2.append((d2.begin, d2.end))

            # begin to search, first move the index
            cnt1 += 1
            cnt2 += 1

            # search
            for i in range(cnt2, len(arr2)):
                diff = arr1[cnt1] - arr2[i]
                if diff < 5:    # match!
                    cnt2 = i
                    d1.begin = cnt1
                    d2.begin = cnt2
                    break

            if i == len(arr2) - 1:  # find through the whole video, but can't match
                cnt1 += 1
                cnt2 += 1
                d1.begin = cnt1
                d2.begin = cnt2


    a = 100
    return com_arr1, com_arr2






if __name__ == '__main__':
    video_path1 = 'D:/wt/test.mp4'      # video changed by boss's wife
    video_path2 = 'D:/wt/test2.mp4'     # yours

    print('Parsing video...')
    hash_arr1 = get_hash_array(video_path1)
    print('Done parsing video: ', video_path1)
    hash_arr2 = get_hash_array(video_path2)
    print('Done parsing video: ', video_path2)

    com_arr1, com_arr2 = compare_hash_arr(hash_arr1, hash_arr2)
    print('------------------------')
    print(len(com_arr1))
    print(com_arr1)
    print(com_arr2)
    a = 0


# cap = cv2.VideoCapture(video_path)
# cap2 = cv2.VideoCapture(video_path2)
# if not cap.isOpened() or not cap2.isOpened():
#     print('Unable to open, ', video_path)
#     exit()
#
#
# cnt = 0
# image1_arr = []
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.resize(frame, (480, 270))
#     frame = Image.fromarray(frame)
#
#     image1_arr.append(imagehash.average_hash(frame))
#
#
#     # cv2.imshow('www', frame)
#     # cv2.waitKey(0)
#
#
# # cv2.imshow('www', frame)
# # cv2.waitKey(0)
# #
# cnt2 = 0
# while True:
#     ret2, frame2 = cap2.read()
#     if not ret2:
#         break
#
#     # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     frame2 = cv2.resize(frame2, (240, 135))
#
#     # frame2 = frame2.astype(np.float)
#     frame2 = Image.fromarray(frame2)
#
#
#
#     t1 = time.time()
#     # frame2 = cv2.resize(frame2, (480, 270))
#     # frame = cv2.resize(frame, (480, 270))
#     ans = calculate_image_similarity(frame, frame2)
#     # ans = sum(sum(sum(abs(frame2 - frame))))
#     # print(time.time() - t1)
#
#     # print(str(cnt2) + ' ' + str(sum(sum(sum(abs(frame2 - frame))))))
#     print(str(cnt2) + ' ' + str(ans))
#     cnt2 += 1