# vggface2
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tensorflow.keras.models import Model

np.random.seed(0)

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
imgdir = f'{drive}DataStorage/CelebA/Img/img_align_celeba/'
identities_fp = f'{drive}DataStorage/CelebA/Anno/identity_CelebA.txt'
model = VGGFace(model='resnet50')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

identities_df = pd.read_csv(identities_fp, header=None, sep=' ')
imageV = np.array(identities_df[0])
identities = {}
for index, row in identities_df.iterrows():
    image_fp = row[0]
    identity = row[1]
    if identity in identities:
        identities[identity].append(image_fp)
    else:
        identities[identity] = [image_fp]

n = 1000  # number of episodes for same and for different identities
print(f'Number of episodes n={n}')
ACC_ED = []
ACC_CS = []
THRESH_ED = []
THRESH_CS = []
for k in range(1, 11):  # number of support set samples - k-shot verification
    image_tuples_same_id = []
    while True:
        identity = np.random.choice(list(identities.keys()), 1)[0]
        if len(identities[identity]) > k:
            tup = np.random.choice(identities[identity], k + 1, replace=False)
            image_tuples_same_id.append(tup)
        if len(image_tuples_same_id) == n:
            break

    image_tuples_diff_id = []
    while True:
        identity = np.random.choice(list(identities.keys()), 2, replace=False)
        if len(identities[identity[0]]) >= k:
            tup1 = np.random.choice(identities[identity[0]], k, replace=False)
            tup2 = np.random.choice(identities[identity[1]], 1, replace=False)
            tup = np.concatenate((tup1, tup2))
            image_tuples_diff_id.append(tup)
        if len(image_tuples_diff_id) == n:
            break
    DS = []
    DD = []
    SS = []
    SD = []
    for i in range(n):
        same_img_1 = [os.path.join(imgdir, x) for x in image_tuples_same_id[i][:-1]]
        same_img_2 = os.path.join(imgdir, image_tuples_same_id[i][-1])
        diff_img_1 = [os.path.join(imgdir, x) for x in image_tuples_diff_id[i][:-1]]
        diff_img_2 = os.path.join(imgdir, image_tuples_diff_id[i][-1])

        faces = np.stack([asarray(Image.fromarray(plt.imread(x)).resize((224, 224))).astype('float32') for x in same_img_1] +
                         [asarray(Image.fromarray(plt.imread(same_img_2)).resize((224, 224))).astype('float32')] +
                         [asarray(Image.fromarray(plt.imread(x)).resize((224, 224))).astype('float32') for x in diff_img_1] +
                         [asarray(Image.fromarray(plt.imread(diff_img_2)).resize((224, 224))).astype('float32')], axis=0)
        faces = preprocess_input(faces, version=2)
        embeddings = model.predict(faces)

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

        support_vector_same = np.mean(embeddings[:k, :], axis=0) if k > 1 else embeddings[0, :]
        query_vector_same = embeddings[k, :]
        support_vector_diff = np.mean(embeddings[k + 1:-1, :], axis=0) if k > 1 else embeddings[2, :]
        query_vector_diff = embeddings[-1, :]

        DS.append(np.linalg.norm(support_vector_same - query_vector_same))
        DD.append(np.linalg.norm(support_vector_diff - query_vector_diff))

        SS.append(np.dot(support_vector_same, query_vector_same) / (np.linalg.norm(support_vector_same) * np.linalg.norm(query_vector_same)))
        SD.append(np.dot(support_vector_diff, query_vector_diff) / (np.linalg.norm(support_vector_diff) * np.linalg.norm(query_vector_diff)))

    threshold_ed = np.arange(0.0, 2.005, 0.005)
    threshold_cs = np.arange(-1.0, 1.005, 0.005)

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
    ACC_ED.append(opt_acc_ed)
    THRESH_ED.append(opt_thres_ed)
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
    ACC_CS.append(opt_acc_cs)
    THRESH_CS.append(opt_thres_cs)
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
plt.plot(np.arange(1, 11), ACC_ED, '.-', label='accuracy (euc.dist.)')
plt.plot(np.arange(1, 11), ACC_CS, '.-', label='accuracy (cos.sim.)')
plt.plot(np.arange(1, 11), THRESH_ED, '.-', label='threshold (euc.dist.)')
plt.plot(np.arange(1, 11), THRESH_CS, '.-', label='threshold (cos.sim.)')
plt.xlabel('k (# support samples)')
plt.grid()
plt.legend()

plt.show()