import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from keras_vggface.utils import preprocess_input
from src.face_recognition import generate_random_image_tuples, feature_extractor, id2fp

np.random.seed(0)  # set random seed

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'  # my second drive where data are stored
imgdir = f'{drive}DataStorage/CelebA/Img/img_align_celeba/'  # images directory
identities_fp = f'{drive}DataStorage/CelebA/Anno/identity_CelebA.txt'  # identities .txt filepath
identities_df = pd.read_csv(identities_fp, header=None, sep=' ')  # identities data frame
n = 1000  # number of episodes for same and for different identities scenarios

print(f'Number of episodes n={n}')

model = feature_extractor()  # load VGGFace feature extraction model
identities = id2fp(identities_df)  # load a dict mapping between identities and image filepaths

ACC_ED = []
ACC_CS = []
THRESH_ED = []
THRESH_CS = []
for k in range(1, 11):  # k: number of support set samples - k-shot verification
    image_tuples_same_id = generate_random_image_tuples(identities, k, n, same_identity=True)
    image_tuples_diff_id = generate_random_image_tuples(identities, k, n, same_identity=False)

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
