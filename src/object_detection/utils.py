import os

import numpy as np
from PIL import Image
import torch
import clip
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def get_anchor(instances, n, model, preprocess, directory, extractor, device):
    categories = {x["id"]: x["name"] for x in instances["categories"]}
    anchor = {}
    for category in categories:
        category_instances = np.random.permutation(
            [
                (i, x)
                for i, x in enumerate(instances["annotations"])
                if x["category_id"] == category
            ]
        )

        g = []
        c = 0
        while len(g) < n:
            i, instance = category_instances[c]
            c += 1
            x, y, w, h = instance["bbox"]

            if w < 120 or h < 120:
                continue

            image = [
                y
                for y in instances["images"]
                if instances["annotations"][i]["image_id"] == y["id"]
            ][0]

            if extractor != "none":
                cropped = (
                    Image.open(f"{directory}/val2014/{image['file_name']}")
                    .crop((x, y, x + w, y + h))
                    .convert("RGB")
                )
                g.append(preprocess(cropped).unsqueeze(0).to(device))
            else:
                g.append(
                    {
                        "image": f"{directory}/val2014/{image['file_name']}",
                        "box": (x, y, x + w, y + h),
                    }
                )

        if extractor != "none":
            with torch.no_grad():
                anchor[category] = model(torch.cat(g, dim=0))
        else:
            anchor[category] = g

    return anchor


def get_model(name, device):
    if name == "clip":
        model, preprocess = clip.load("ViT-L/14", device=device)

        def model_func(x):
            return model.encode_image(x)

        return model_func, preprocess
    elif name == "vit":
        model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            num_classes=0,
        ).to(device)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        return model, transform
    elif name == "none":
        return None, None


def get_uc_sets(directory):
    anchor = []
    for fn in os.listdir(f"{directory}/anchor"):
        anchor.append(f"{directory}/anchor/{fn}")

    positive = []
    for fn in os.listdir(f"{directory}/test"):
        positive.append(f"{directory}/test/{fn}")

    negative = []
    for fn in os.listdir(f"{os.path.dirname(directory)}/other"):
        negative.append(f"{os.path.dirname(directory)}/other/{fn}")

    return anchor, positive, negative
