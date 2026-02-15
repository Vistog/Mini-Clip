from __future__ import annotations

import math
import torch
import torch.nn as nn

from mini_clip.models.components import *  # noqa: F401,F403
from mini_clip.registry import get
from mini_clip.models.clip import MiniCLIP


def build_model(cfg: dict) -> MiniCLIP:
    mcfg = cfg["model"]

    # 1) энкодеры
    ImageEnc = get("image_encoder", mcfg["image_encoder"]["name"])
    TextEnc = get("text_encoder", mcfg["text_encoder"]["name"])

    image_encoder = ImageEnc(**mcfg["image_encoder"])
    text_encoder = TextEnc(**mcfg["text_encoder"])

    # 2) projection heads
    proj_cfg = mcfg["projection"]
    Proj = get("projection", proj_cfg["type"])

    image_proj = Proj(in_dim=image_encoder.out_dim, out_dim=mcfg["embed_dim"], **proj_cfg)
    text_proj = Proj(in_dim=text_encoder.out_dim, out_dim=mcfg["embed_dim"], **proj_cfg)

    # 3) temperature / logit scale
    tcfg = mcfg["temperature"]
    init_temp = float(tcfg["init"])

    init_logit_scale = math.log(1.0 / init_temp)
    logit_scale = nn.Parameter(torch.tensor(init_logit_scale), requires_grad=bool(tcfg["learnable"]))

    return MiniCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_proj=image_proj,
        text_proj=text_proj,
        logit_scale=logit_scale,
    )
