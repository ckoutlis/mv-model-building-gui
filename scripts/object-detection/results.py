import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

directory = "/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/DataStorage/coco"
instances = json.load(open(f"{directory}/annotations/instances_val2014.json"))
extractors = ["clip", "vit", "none"]
ucs = ["logo", "flags", "svastika"]
categories = {x["id"]: x["name"] for x in instances["categories"][:16]}

with open("../../results/object-detection/eval/metrics_coco.pickle", "rb") as f:
    metrics = pickle.load(f)

for query_type in ["image", "boxes"]:
    for extractor in extractors:
        c = 0
        plt.figure(figsize=(12, 12))
        plt.suptitle(f"{query_type} - {extractor}")
        for category in metrics[extractor]:
            c += 1
            xy = np.array(
                [
                    (
                        metrics[extractor][category][x]["cover"],
                        np.max(
                            metrics[extractor][category][x]["similarity"][query_type]
                        )
                        if extractor != "none"
                        else np.max(
                            metrics[extractor][category][x]["similarity"][query_type][
                                "uqi"
                            ]
                        ),
                    )
                    for x in metrics[extractor][category]
                    if (
                        extractor != "none"
                        and metrics[extractor][category][x]["similarity"][query_type]
                    )
                    or (
                        (
                            extractor == "none"
                            and metrics[extractor][category][x]["similarity"][
                                query_type
                            ]["uqi"]
                        )
                    )
                ]
            )
            plt.subplot(4, 4, c)
            cover_index = xy[:, 0] > 0.0
            plt.plot(
                xy[np.logical_not(cover_index), 0],
                xy[np.logical_not(cover_index), 1],
                "xr",
            )
            plt.plot(xy[cover_index, 0], xy[cover_index, 1], "ob")
            plt.title(categories[category])
        plt.savefig(
            f"../../results/object-detection/figs/{query_type} - {extractor}.png"
        )

acc_image = pd.DataFrame(
    data=np.full((len(categories), len(extractors)), np.nan),
    index=[categories[category] for category in categories],
    columns=extractors,
)
acc_boxes = pd.DataFrame(
    data=np.full((len(categories), len(extractors)), np.nan),
    index=[categories[category] for category in categories],
    columns=extractors,
)
acc_both = pd.DataFrame(
    data=np.full((len(categories), len(extractors)), np.nan),
    index=[categories[category] for category in categories],
    columns=extractors,
)
threshold = pd.DataFrame(
    data=np.full((len(categories), len(extractors)), np.nan),
    index=[categories[category] for category in categories],
    columns=extractors,
)
for extractor in extractors:
    for category in categories:
        labels = [
            metrics[extractor][category][x]["cover"] > 0.0
            for x in metrics[extractor][category]
        ]
        scores = [
            [
                metrics[extractor][category][x]["similarity"]["image"],
                metrics[extractor][category][x]["similarity"]["boxes"],
            ]
            if extractor != "none"
            else [
                [
                    metrics[extractor][category][x]["similarity"]["image"]["rmse"],
                    metrics[extractor][category][x]["similarity"]["image"]["uqi"],
                ],
                [
                    metrics[extractor][category][x]["similarity"]["boxes"]["rmse"],
                    metrics[extractor][category][x]["similarity"]["boxes"]["uqi"],
                ],
            ]
            for x in metrics[extractor][category]
        ]
        acc = []
        for thres in np.linspace(0, 1, 20):
            preds = [
                max(x[0]) > thres if extractor != "none" else max(x[0][1]) > thres
                for x in scores
            ]
            acc_im = balanced_accuracy_score(labels, preds)
            preds = [
                max(x[1] if x[1] else [0.0]) > thres
                if extractor != "none"
                else max(x[1][1] if x[1][1] else [0.0]) > thres
                for x in scores
            ]
            acc_bx = balanced_accuracy_score(labels, preds)
            preds = [
                max(x[0] + x[1]) > thres
                if extractor != "none"
                else max(x[0][1] + x[1][1]) > thres
                for x in scores
            ]
            acc_2 = balanced_accuracy_score(labels, preds)
            acc.append((thres, acc_im, acc_bx, acc_2))
        acc_image[extractor][categories[category]] = max([x[1] for x in acc])
        acc_boxes[extractor][categories[category]] = max([x[2] for x in acc])
        acc_both[extractor][categories[category]] = max([x[3] for x in acc])
        threshold[extractor][categories[category]] = acc[
            np.argmax([x[3] for x in acc])
        ][0]

print("Image:")
print(acc_image.round(3))
print(acc_image.mean(axis=0).round(3))
print("\nBoxes:")
print(acc_boxes.round(3))
print(acc_boxes.mean(axis=0).round(3))
print("\nBoth:")
print(acc_both.round(3))
print(acc_both.mean(axis=0).round(3))
print("\nThreshold:")
print(threshold.round(3))
print(threshold.mean(axis=0).round(3))

with open("../../results/object-detection/eval/metrics_uc.pickle", "rb") as f:
    metrics = pickle.load(f)

acc_uc = pd.DataFrame(
    data=np.zeros((len(ucs), len(extractors))),
    index=ucs,
    columns=extractors,
)
threshold_uc = pd.DataFrame(
    data=np.zeros((len(ucs), len(extractors))),
    index=ucs,
    columns=extractors,
)
for extractor in extractors:
    # t = threshold.mean(axis=0)[extractor]
    for uc in ucs:
        acc = []
        for t in np.linspace(0, 1, 20):
            samples = metrics[extractor][uc]
            labels = ["p" in x for x in samples]
            preds = [max(samples[x]) > t for x in samples]
            acc.append((t, balanced_accuracy_score(labels, preds)))
        acc_uc[extractor][uc] = max([x[1] for x in acc])
        threshold_uc[extractor][uc] = acc[np.argmax([x[1] for x in acc])][0]

print("\nUse Cases accuracy:")
print(acc_uc.round(3))
print(acc_uc.mean(axis=0).round(3))
print("\nUse Cases threshold:")
print(threshold_uc.round(3))
print(threshold_uc.mean(axis=0).round(3))
