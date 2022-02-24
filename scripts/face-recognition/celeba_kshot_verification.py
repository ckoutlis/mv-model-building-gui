import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.face_recognition import *

np.random.seed(0)  # set random seed

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'  # my second drive where data are stored
imgdir = f'{drive}DataStorage/CelebA/Img/img_align_celeba/'  # images directory
identities_fp = f'{drive}DataStorage/CelebA/Anno/identity_CelebA.txt'  # identities .txt filepath
identities_df = pd.read_csv(identities_fp, header=None, sep=' ')  # identities data frame
n = 1000  # number of episodes for same and for different identities scenarios
threshold_ed = np.arange(0.0, 2.005, 0.005)
threshold_cs = np.arange(-1.0, 1.005, 0.005)

print(f'Number of episodes n={n}')

model = feature_extractor()  # load VGGFace feature extraction model
identities = id2fp(identities_df)  # load a dict mapping between identities and image filepaths

# ACC_ED, ACC_CS, THRESH_ED, THRESH_CS = [], [], [], []
accuracy_distance, accuracy_similarity, threshold_distance, threshold_similarity = [], [], [], []
for k in range(1, 11):  # k: number of support set samples - k-shot verification
    image_tuples_same_id = generate_random_image_tuples(identities, k, n, same_identity=True)
    image_tuples_diff_id = generate_random_image_tuples(identities, k, n, same_identity=False)

    DS, DD, SS, SD = [], [], [], []
    for i in range(n):
        faces = np.concatenate((get_faces(imgdir, image_tuples_same_id[i]),
                                get_faces(imgdir, image_tuples_diff_id[i])),
                               axis=0)
        support_same, query_same, support_diff, query_diff = get_embeddings(model, faces, k)

        DS.append(euclidean_distance(support_same, query_same))
        DD.append(euclidean_distance(support_diff, query_diff))
        SS.append(cosine_similarity(support_same, query_same))
        SD.append(cosine_similarity(support_diff, query_diff))

    accuracy = []
    for t in threshold_ed:
        same = (np.array(DS) < t).astype(int)
        diff = (np.array(DD) < t).astype(int)
        TP = np.sum(same == 1)
        TN = np.sum(diff == 0)
        FP = np.sum(diff == 1)
        FN = np.sum(same == 0)
        accuracy.append((TP + TN) / (TP + TN + FP + FN))
    indx = np.argmax(accuracy)
    opt_acc_ed = accuracy[indx]
    opt_thres_ed = threshold_ed[indx]
    accuracy_distance.append(opt_acc_ed)
    threshold_distance.append(opt_thres_ed)
    print(f'Euclidean distance[k={k}]: best accuracy {opt_acc_ed * 100:1.1f}% for threshold {opt_thres_ed:1.3f}')

    accuracy = []
    for t in threshold_cs:
        same = (np.array(SS) > t).astype(int)
        diff = (np.array(SD) > t).astype(int)
        TP = np.sum(same == 1)
        TN = np.sum(diff == 0)
        FP = np.sum(diff == 1)
        FN = np.sum(same == 0)
        accuracy.append((TP + TN) / (TP + TN + FP + FN))
    indx = np.argmax(accuracy)
    opt_acc_cs = accuracy[indx]
    opt_thres_cs = threshold_cs[indx]
    accuracy_similarity.append(opt_acc_cs)
    threshold_similarity.append(opt_thres_cs)
    print(f'Cosine similarity[k={k}]: best accuracy {opt_acc_cs * 100:1.1f}% for threshold {opt_thres_cs:1.3f}\n')

    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    plt.title(f'Euclidean distance - acc={opt_acc_ed * 100:1.1f}%, thresh={opt_thres_ed:1.3f}\n k={k}, n={n}')
    y1, _, _ = plt.hist(DS, alpha=0.8, label='same identity')
    y2, _, _ = plt.hist(DD, alpha=0.8, label='diff. identity')
    plt.plot([opt_thres_ed, opt_thres_ed], [0, np.max(np.concatenate((y1, y2)))], '-r', linewidth=1.5, label='optimal threshold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f'Cosine similarity - acc={opt_acc_cs * 100:1.1f}%, thresh={opt_thres_cs:1.3f}\n k={k}, n={n}')
    y1, _, _ = plt.hist(SS, alpha=0.8, label='same identity')
    y2, _, _ = plt.hist(SD, alpha=0.8, label='diff. identity')
    plt.plot([opt_thres_cs, opt_thres_cs], [0, np.max(np.concatenate((y1, y2)))], '-r', linewidth=1.5, label='optimal threshold')
    plt.legend()

plt.figure()
plt.plot(np.arange(1, 11), accuracy_distance, '.-', label='accuracy (euc.dist.)')
plt.plot(np.arange(1, 11), accuracy_similarity, '.-', label='accuracy (cos.sim.)')
plt.plot(np.arange(1, 11), threshold_distance, '.-', label='threshold (euc.dist.)')
plt.plot(np.arange(1, 11), threshold_similarity, '.-', label='threshold (cos.sim.)')
plt.xlabel('k (# support samples)')
plt.grid()
plt.legend()

plt.show()
