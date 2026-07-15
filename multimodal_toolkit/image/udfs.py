"""图片分析管道的 Daft UDF。

每张图片输出一个包含原始测量值的 struct（不含结论——结论由
rules.add_rule_columns 稍后派生）：

  width, height    —— 原始像素尺寸（缩放之前的）。
  face_count       —— SCRFD 检测到的人脸数量（粗筛阈值 FACE_DET_THRESH 之上的）。
  face_score       —— 最大人脸的检测置信度，[0, 1]；没有脸时为 0.0。
  face_area_ratio  —— 最大人脸框面积 / 整图面积，[0, 1]；直观理解是
                     "脸占了画面多大"（0.2 ≈ 大头照，0.001 ≈ 人群中的小脸）。
  blur_score       —— 整图清晰度（Laplacian 方差，越高越清晰；
                     原理见 quality.py）。
  face_blur_score  —— 同样的清晰度指标，但只算最大人脸的裁剪区域；
                     没检测到脸时为 null。

所有 face 派生指标（face_score / face_area_ratio / face_blur_score）都
取自同一张脸——面积最大的那张。规则引擎把它们 AND 在一起时，判定的
才是"同一张脸"；如果分别取自不同的脸（比如置信度取全局最高分），
一个大面积低置信度的误检加一个极小的高置信度脸就能拼出假的 has_face。
已知局限：大误检和中等大小的真脸同框时，以最大脸为准会漏判——
需要彻底解决时应落 per-face 列表列，让规则做"存在任一张脸满足"的判断。
"""
from __future__ import annotations

import daft

IMAGE_QUALITY_DTYPE = daft.DataType.struct(
    {
        "width": daft.DataType.int32(),
        "height": daft.DataType.int32(),
        "face_count": daft.DataType.int32(),
        "face_score": daft.DataType.float64(),
        "face_area_ratio": daft.DataType.float64(),
        "blur_score": daft.DataType.float64(),
        "face_blur_score": daft.DataType.float64(),  # 没检测到脸时为 null
    }
)

# 下载失败（输入为 null）或解码失败的图片返回这个全 null 行；
# 下游据此生成 status 列，而不是把行丢掉。
_NULL_ROW = {
    "width": None,
    "height": None,
    "face_count": None,
    "face_score": None,
    "face_area_ratio": None,
    "blur_score": None,
    "face_blur_score": None,
}


@daft.cls(cpus=1, max_retries=2)
class ImageQualityUDF:
    def __init__(self) -> None:
        # 每个 worker 进程只加载一次模型，而不是每行加载——模型初始化
        # 才是开销大头（与音频 AsrUDF 的模式相同）。
        from multimodal_toolkit.image.detector import get_detector

        self._detector = get_detector()

    @daft.method.batch(return_dtype=IMAGE_QUALITY_DTYPE)
    def __call__(self, image_bytes_col):
        from multimodal_toolkit.image.quality import (
            crop_bbox,
            decode_image,
            laplacian_variance,
            resize_long_edge,
        )

        results = []
        for image_bytes in image_bytes_col.to_pylist():
            img = decode_image(image_bytes)
            if img is None:
                results.append(dict(_NULL_ROW))
                continue
            height, width = img.shape[:2]
            # 人脸检测和清晰度都在缩放后的图上做：坐标系统一，
            # 且清晰度阈值在不同分辨率之间保持可比。
            resized, _ = resize_long_edge(img)
            faces = self._detector.detect(resized)

            face_score = 0.0
            face_blur_score = None
            face_area_ratio = 0.0
            if faces:
                # faces[0] 是面积最大的框（检测器已按面积排序）——用最显眼
                # 的那张脸代表这张图"的"脸，三个指标都从它身上取。
                largest = faces[0]
                face_score = largest["score"]
                x1, y1, x2, y2 = largest["bbox"]
                rh, rw = resized.shape[:2]
                face_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / (rw * rh)
                face_crop = crop_bbox(resized, largest["bbox"])
                if face_crop is not None:
                    face_blur_score = laplacian_variance(face_crop)

            results.append(
                {
                    "width": width,
                    "height": height,
                    "face_count": len(faces),
                    "face_score": face_score,
                    "face_area_ratio": face_area_ratio,
                    "blur_score": laplacian_variance(resized),
                    "face_blur_score": face_blur_score,
                }
            )
        return results
