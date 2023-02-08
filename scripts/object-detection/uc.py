import json
import torch
from PIL import Image
import numpy as np
import pickle
from sewar.full_ref import uqi
from src.object_detection.utils import get_model, get_uc_sets

device = "cuda:0"
ucs = ["logo", "flags", "svastika"]
extractors = ["clip", "vit", "none"]
metrics = {x: {} for x in extractors}
for uc in ucs:
    directory = (
        f"/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/DataStorage/model-building-uc/{uc}"
    )
    for extractor in extractors:
        model, preprocess = get_model(extractor, device)
        anchor, positive, negative = get_uc_sets(directory)
        metrics[extractor][uc] = {}

        # Similarity estimation between anchors and the whole current image
        test = [(x, "p") for x in positive] + [(x, "n") for x in negative]
        for i, test_ in enumerate(test):
            metrics[extractor][uc][f"{test_[1]}{i}"] = []
            for j, anchor_ in enumerate(anchor):
                print(
                    f"\r{uc}-{extractor}-test:{i+1}/{len(test)}-anchor:{j+1}/{len(anchor)}",
                    end="",
                )
                if extractor != "none":
                    with torch.no_grad():
                        test_features = model(
                            (
                                preprocess(Image.open(test_[0]).convert("RGB"))
                                .unsqueeze(0)
                                .to(device)
                            )
                        )
                        anchor_features = model(
                            (
                                preprocess(Image.open(anchor_).convert("RGB"))
                                .unsqueeze(0)
                                .to(device)
                            )
                        )

                    metrics[extractor][uc][f"{test_[1]}{i}"].append(
                        (
                            torch.sum(test_features * anchor_features, dim=1)
                            / (test_features.norm() * anchor_features.norm())
                        ).tolist()[0]
                    )
                else:
                    metrics[extractor][uc][f"{test_[1]}{i}"].append(
                        uqi(
                            np.array(
                                Image.open(anchor_).convert("RGB").resize((224, 224))
                            ),
                            np.array(
                                Image.open(test_[0]).convert("RGB").resize((224, 224))
                            ),
                        )
                    )

                with open(f"../../results/object-detection/eval/metrics_uc.pickle", "wb") as h:
                    pickle.dump(metrics, h, protocol=pickle.HIGHEST_PROTOCOL)
