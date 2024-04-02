import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pandas as pd

def read_file_list(filename):

    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    return list

def AKAZE_Match_Score(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = flann.knnMatch(des1, des2, k=2)

    # matchesMask = [[0, 0] for i in range(len(matches))]

    score = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            score += 1
            # matchesMask[i] = [1, 0]

    # drawParams = dict(matchColor=(0, 255, 0),
    #                   singlePointColor=(255, 0, 0),
    #                   matchesMask=matchesMask,
    #                   flags=0
    #                   )
    # resultImage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawParams)
    # plt.xticks([]), plt.yticks([])
    # plt.imshow(resultImage), plt.show()

    score = score/len(matches)
    # print(score)
    return score


def get_file_name(file_path):
    file_names = os.listdir(file_path)
    return file_names


def calc_pr(score_list, threshold):
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(score_list)):
        gt = 0
        pred = 0
        if (score_list[i][0] % 2) == 1:
            if score_list[i][1] == (score_list[i][0]+1):
                gt = 1
        else:
            if score_list[i][1] == (score_list[i][0]-1):
                gt = 1

        if score_list[i][2] >= threshold:
            pred = 1


        if gt == 1 and pred == 1:
            TP += 1
        if gt == 0 and pred == 1:
            FP += 1
        if gt == 1 and pred == 0:
            FN += 1

    if TP != 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    else:
        precision = 0
        recall = 0

    return precision, recall
    


if __name__ == '__main__':
    dir_path = 'loop-dataset'
    pic_names = get_file_name(dir_path)
    score_list = []

    for i in range(len(pic_names)):
       pic_path1 = os.path.join(dir_path, pic_names[i])
       id1, suffix1 = os.path.splitext(pic_names[i])
       for j in range(i+1, len(pic_names)):
           pic_path2 = os.path.join(dir_path, pic_names[j])
           id2, suffix2 = os.path.splitext(pic_names[j])
           score = AKAZE_Match_Score(pic_path1, pic_path2)
           sq = [int(id1), int(id2), score]
           print(id1, ",", id2, ":", score)
           score_list.append(sq)

    sample_size = 100

    prs = np.zeros(sample_size)
    res = np.zeros(sample_size)

    ths = np.linspace(0, 0.15, sample_size)
    idx = 0
    for th in ths:
        precision, recall = calc_pr(score_list, th)
        prs[idx] = precision
        res[idx] = recall
        idx += 1

        print("[", precision, ",", recall, "]")

    plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')

    l1, = plt.plot(res, prs)
    # plt.savefig(fname='pr.png')
    # ls = read_file_list('pr.txt')
    # prs2 = np.zeros(len(ls))
    # res2 = np.zeros(len(ls))
    # for i in range(len(ls)):
    #     prs2[i] = ls[i][0]
    #     res2[i] = ls[i][1]
    # d = {'x': res2, 'y': prs2}
    # df = pd.DataFrame(data=d)
    # df = df.sort_values(by=['x', 'y'])
    #
    # l2, = plt.plot(df['x'], df['y'], 'b-.')
    plt.legend(handles=(l1), labels=['Ours'])
    plt.savefig(fname='pr2.png')
    plt.show()
