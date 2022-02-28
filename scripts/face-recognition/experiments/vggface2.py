import pandas as pd
import pickle
from src.face_recognition.experiment import *

np.random.seed(0)  # set random seed

basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # mv-model-building-gui absolute path
drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'  # my second drive where data are stored
imgdir = f'{drive}DataStorage/VGG-Face2/data/test/'  # images directory
identities_fp = f'{drive}DataStorage/VGG-Face2/meta/identity_meta.csv'  # identities .csv filepath
identities_meta_csv = pd.read_csv(identities_fp, sep=',')  # identities meta csv data frame
identities_list = []
for name, class_id, flag in zip(identities_meta_csv[' Name'], identities_meta_csv['Class_ID'], identities_meta_csv[' Flag']):
    if not flag:
        filepaths = os.listdir(f'{imgdir}{class_id}')
        identities_list.extend([[f'{imgdir}{class_id}/{filepath}', name] for filepath in filepaths])
identities_df = pd.DataFrame(identities_list)
K = 20  # maximum number of shots
n = 1000  # number of episodes for same and for different identities scenarios
threshold_distance_range = np.arange(0.0, 2.005, 0.005)  # candidate distance thresholds for same vs. different identity
threshold_similarity_range = np.arange(-1.0, 1.005, 0.005)  # candidate similarity thresholds for same vs. different identity
savfig = True  # whether to save the figures
saveval = True  # whether to save accuracy and thresholds

print(f'Number of episodes n={n}')

model = feature_extractor()  # load VGGFace feature extraction model
identities = id2fp(identities_df)  # load a dict mapping between identities and image filepaths
plot_statistics(f'{basepath}/results/face-recognition/figs/VGGFace2_statistics.jpg', identities, identities_df, savfig)

# optimal accuracy and threshold for each k (using distance and similarity metrics respectively)
accuracy_distance, accuracy_similarity, threshold_distance, threshold_similarity = {}, {}, {}, {}
for k in range(1, K + 1):  # k: number of support set samples - k-shot verification
    image_tuples_same_id = generate_random_image_tuples(identities, k, n, same_identity=True)
    image_tuples_diff_id = generate_random_image_tuples(identities, k, n, same_identity=False)

    distance_same, distance_different, similarity_same, similarity_different = [], [], [], []
    for i in range(n):
        faces = np.concatenate((get_faces('', image_tuples_same_id[i]),
                                get_faces('', image_tuples_diff_id[i])),
                               axis=0)
        support_same, query_same, support_diff, query_diff = get_embeddings(model, faces, k)

        distance_same.append(euclidean_distance(support_same, query_same))
        distance_different.append(euclidean_distance(support_diff, query_diff))
        similarity_same.append(cosine_similarity(support_same, query_same))
        similarity_different.append(cosine_similarity(support_diff, query_diff))

    optimal_accuracy_distance, optimal_threshold_distance = evaluation(
        thresholds=threshold_distance_range,
        metric_same=distance_same,
        metric_different=distance_different,
        mode='distance',
        k=k,
        showtxt=True
    )
    accuracy_distance[k] = optimal_accuracy_distance
    threshold_distance[k] = optimal_threshold_distance

    optimal_accuracy_similarity, optimal_threshold_similarity = evaluation(
        thresholds=threshold_similarity_range,
        metric_same=similarity_same,
        metric_different=similarity_different,
        mode='similarity',
        k=k,
        showtxt=True
    )
    accuracy_similarity[k] = optimal_accuracy_similarity
    threshold_similarity[k] = optimal_threshold_similarity

    if saveval:
        with open(os.path.join(basepath, 'results/face-recognition/eval/VGGFace2_accuracy_distance.pickle'), 'wb') as h:
            pickle.dump(accuracy_distance, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(basepath, 'results/face-recognition/eval/VGGFace2_accuracy_similarity.pickle'), 'wb') as h:
            pickle.dump(accuracy_similarity, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(basepath, 'results/face-recognition/eval/VGGFace2_threshold_distance.pickle'), 'wb') as h:
            pickle.dump(threshold_distance, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(basepath, 'results/face-recognition/eval/VGGFace2_threshold_similarity.pickle'), 'wb') as h:
            pickle.dump(threshold_similarity, h, protocol=pickle.HIGHEST_PROTOCOL)

    plot_histograms(
        f'{basepath}/results/face-recognition/figs/VGGFace2_k{k}.jpg',
        optimal_accuracy_distance,
        optimal_threshold_distance,
        optimal_accuracy_similarity,
        optimal_threshold_similarity,
        distance_same,
        distance_different,
        similarity_same,
        similarity_different,
        k,
        n,
        savfig
    )

plot_evaluation(
    f'{basepath}/results/face-recognition/figs/VGGFace2_evaluation.jpg',
    [accuracy_distance[x] for x in accuracy_distance],
    [accuracy_similarity[x] for x in accuracy_similarity],
    [threshold_distance[x] for x in threshold_distance],
    [threshold_similarity[x] for x in threshold_similarity],
    savfig
)
