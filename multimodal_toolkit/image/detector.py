"""基于 InsightFace SCRFD 的人脸检测。

给非 CV 背景读者：SCRFD 是一个小型神经网络，输入一张图片，对找到的
每张人脸返回一个包围框（框住脸的矩形）和一个 [0, 1] 的检测置信度——
即"这个矩形里真的是一张脸"的把握有多大。它不做身份识别（"这是谁"），
只回答"脸在哪里"。我们刻意只加载检测模块，跳过 InsightFace 的识别/
关键点模型，启动和推理都轻量得多。
"""
from __future__ import annotations

from . import config


class FaceDetector:
    """InsightFace SCRFD 人脸检测器（仅检测模块，CPU 推理）。"""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        kwargs: dict = {
            # 模型包名，如 "buffalo_l"。首次使用时自动下载到 ~/.insightface，
            # 除非用 INSIGHTFACE_ROOT 指向预置好的模型目录（离线/容器场景）。
            "name": config.INSIGHTFACE_MODEL,
            "allowed_modules": ["detection"],
            "providers": ["CPUExecutionProvider"],
        }
        if config.INSIGHTFACE_ROOT:
            kwargs["root"] = config.INSIGHTFACE_ROOT
        self._app = FaceAnalysis(**kwargs)
        # ctx_id=-1 表示用 CPU。det_size 是检测前图片在内部被缩放到的正方形
        # 分辨率——越大能检出越小的脸但越慢；640 是 SCRFD 默认的平衡点。
        # det_thresh 是检测器的粗筛阈值，必须显式设置且明显低于业务阈值
        # FACE_DET_SCORE_MIN：业务判定完全归规则引擎所有，检测器只负责
        # 把候选留下来。否则低于 SCRFD 默认值 0.5 的脸根本不会落表，
        # 后续用 SQL 下调业务阈值也找不回来。
        self._app.prepare(
            ctx_id=-1,
            det_thresh=config.FACE_DET_THRESH,
            det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE),
        )

    def detect(self, img_bgr) -> list[dict]:
        """返回人脸列表 [{"bbox": (x1, y1, x2, y2), "score": float}]，按框面积降序。

        bbox 是输入图片的像素坐标：(x1, y1) 左上角，(x2, y2) 右下角。
        score 是检测置信度，取值 [0, 1]。按面积排序让最显眼的脸排在
        最前——下游代码把 faces[0] 当作这张图"的"那张脸。
        """
        faces = self._app.get(img_bgr)
        results = []
        for face in faces:
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            results.append({"bbox": (x1, y1, x2, y2), "score": float(face.det_score)})
        results.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        return results


_DETECTOR: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector()
    return _DETECTOR
