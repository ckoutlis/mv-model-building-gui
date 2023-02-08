import json
import torch
from PIL import Image
import numpy as np
import pickle
from sewar.full_ref import rmse, uqi
from src.object_detection.utils import get_model, get_anchor

directory = "/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/DataStorage/coco"
instances = json.load(open(f"{directory}/annotations/instances_val2014.json"))
categories = {x["id"]: x["name"] for x in instances["categories"][:16]}
device = "cuda:0"
n = 10  # number of prototypes
methods = ["rmse", "uqi"]
extractors = ["clip", "vit", "none"]  # "clip", "vit", "none"
images_per_category = 1000
metrics = {x: {} for x in extractors}
for extractor in extractors:
    np.random.seed(42)
    model, preprocess = get_model(extractor, device)
    anchor = get_anchor(instances, n, model, preprocess, directory, extractor, device)
    for category in categories:
        metrics[extractor][category] = {}
        for ii, image in enumerate(
            np.random.permutation(instances["images"])[:images_per_category].tolist()
        ):
            print(
                f"\r[{extractor}] {category}: {categories[category]}, image: {ii+1}/{images_per_category}",
                end="",
            )

            # Image information
            image_id = image["id"]
            width = image["width"]
            height = image["height"]
            image_objects = [
                (x["category_id"], x["bbox"])
                for x in instances["annotations"]
                if x["image_id"] == image_id
            ]

            # Cover: sum of area fraction depicting instances of the category
            metrics[extractor][category][image_id] = {}
            if category in [x[0] for x in image_objects]:
                metrics[extractor][category][image_id]["cover"] = sum(
                    [x[1][2] * x[1][3] for x in image_objects if x[0] == category]
                ) / (width * height)
            else:
                metrics[extractor][category][image_id]["cover"] = 0.0

            # Similarity estimation between anchors and the whole current image
            metrics[extractor][category][image_id]["similarity"] = {}
            if extractor != "none":
                g = (
                    preprocess(
                        Image.open(f"{directory}/val2014/{image['file_name']}").convert(
                            "RGB"
                        )
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    img_emb = model(g)

                metrics[extractor][category][image_id]["similarity"]["image"] = (
                    torch.sum(anchor[category] * img_emb, dim=1)
                    / (anchor[category].norm() * img_emb.norm())
                ).tolist()
            else:
                g = (
                    Image.open(f"{directory}/val2014/{image['file_name']}")
                    .convert("RGB")
                    .resize((224, 224))
                )
                metrics[extractor][category][image_id]["similarity"]["image"] = {
                    method: [] for method in methods
                }
                for anchor_ in anchor[category]:
                    g0 = (
                        Image.open(anchor_["image"])
                        .convert("RGB")
                        .crop(anchor_["box"])
                        .resize((224, 224))
                    )
                    for method in methods:
                        metrics[extractor][category][image_id]["similarity"]["image"][
                            method
                        ].append(eval(f"{method}(np.array(g0), np.array(g))"))

            # Similarity estimation between anchors and bounding boxes of the current image
            similarity = (
                [] if extractor != "none" else {method: [] for method in methods}
            )
            for _, box in image_objects:
                if extractor != "none":
                    try:
                        g = (
                            preprocess(
                                Image.open(
                                    f"{directory}/val2014/{image['file_name']}"
                                ).convert("RGB").crop(
                                    (box[0], box[1], box[0] + box[2], box[1] + box[3])
                                )
                            )
                            .unsqueeze(0)
                            .to(device)
                        )
                        with torch.no_grad():
                            bx_emb = model(g)
                        similarity.extend(
                            (
                                torch.sum(anchor[category] * bx_emb, dim=1)
                                / (anchor[category].norm() * bx_emb.norm())
                            ).tolist()
                        )
                    except:
                        pass
                else:
                    g = (
                        Image.open(f"{directory}/val2014/{image['file_name']}")
                        .convert("RGB")
                        .crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                        .resize((224, 224))
                    )
                    for anchor_ in anchor[category]:
                        g0 = (
                            Image.open(anchor_["image"])
                            .convert("RGB")
                            .crop(anchor_["box"])
                            .resize((224, 224))
                        )
                        for method in methods:
                            similarity[method].append(
                                eval(f"{method}(np.array(g0), np.array(g))")
                            )

            metrics[extractor][category][image_id]["similarity"]["boxes"] = similarity

            with open(f"../../results/object-detection/eval/metrics_coco.pickle", "wb") as h:
                pickle.dump(metrics, h, protocol=pickle.HIGHEST_PROTOCOL)
