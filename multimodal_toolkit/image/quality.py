"""纯 OpenCV/numpy 的图片质量工具函数（不依赖 Daft，不依赖模型）。

给非 CV 背景读者的清晰度入门：清晰的图片在物体边缘处亮度变化剧烈，
模糊的图片则变化平缓。Laplacian（拉普拉斯）算子对这种剧烈变化敏感，
因此整张图 Laplacian 结果的*方差*就是一个廉价的清晰度分数——
方差高 = 清晰，方差低 = 模糊。该分数本身没有单位，只有在"相同尺寸
的图片之间比较"时才有意义，所以调用方必须先用 resize_long_edge
把图片缩放到固定长边，再计算分数、再和阈值比较。
"""
from __future__ import annotations

import numpy as np

from . import config


def decode_image(image_bytes: bytes | None):
    """把编码后的字节流（jpg/png/...）解码为 BGR 像素矩阵；解码失败返回 None。

    BGR（蓝-绿-红通道顺序）是 OpenCV 的原生格式——本模块的所有函数
    和 SCRFD 检测器都以此为输入。
    """
    import cv2

    if not image_bytes:
        return None
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def resize_long_edge(img, long_edge: int | None = None):
    """缩放图片使长边等于 long_edge（只缩小不放大）。返回 (缩放后图片, 缩放比例)。

    先统一尺寸再打分，才能让清晰度分数跨分辨率可比：同一张照片在
    4000px 和 1000px 下的 Laplacian 方差差异巨大，不归一化就无法用
    一个阈值判断。小图保持原样——放大不会凭空产生细节，反而会扭曲分数。
    """
    import cv2

    long_edge = long_edge or config.IMAGE_LONG_EDGE
    h, w = img.shape[:2]
    edge = max(h, w)
    if edge <= long_edge:
        return img, 1.0
    scale = long_edge / edge
    # INTER_AREA 是缩小图片时推荐的插值方式（锯齿最少）。
    resized = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def laplacian_variance(img_bgr) -> float:
    """清晰度分数：灰度图上 Laplacian 结果的方差。

    越高越清晰。颜色不携带清晰度信息，所以先转灰度；用 CV_64F 保留
    负的边缘响应——无符号类型会把负值截断为 0，导致方差被低估。
    """
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_bbox(img, bbox):
    """按 (x1, y1, x2, y2) 裁剪，坐标钳制到图片边界内；框无效时返回 None。

    检测框可能略微超出图片边界（比如脸贴着画面边缘），所以对坐标做
    钳制而不是直接拒绝。
    """
    h, w = img.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]
