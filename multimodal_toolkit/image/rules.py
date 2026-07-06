"""阈值规则引擎：把原始分数列转成布尔结论列。

模型（人脸检测器、清晰度打分）只输出数字；本模块用 config 里的阈值
（均可通过环境变量覆盖）把数字变成是/否的结论：

  has_face        —— 最大人脸的置信度不低于 FACE_DET_SCORE_MIN，且框面积
                     占整图比例不低于 MIN_FACE_RATIO（两个条件针对同一张
                     脸，用来过滤背景小脸和低置信度误检）。
  is_blurry       —— 整图清晰度低于 BLUR_THRESHOLD。
  is_face_blurry  —— 图里有脸，但人脸区域的清晰度低于 FACE_BLUR_THRESHOLD
                     （背景清晰但脸糊的图能通过 is_blurry，却会在这里被抓住）。

输入必须带 status 列（见 image/workflow/analyze.py）：只有 status = "ok" 的行
才产生结论；下载/解码失败的行结论为 null——"不知道"必须和"判定为否"
区分开，否则下游会把打不开的图当成合规的无人脸图。

原始分数和布尔结论都会落表，因此调阈值只需重跑这一步（或直接写 SQL），
不需要重跑模型。注意 SQL 可调范围的下限是检测器粗筛阈值 FACE_DET_THRESH
——低于它的人脸根本不在表里（见 detector.py）。
"""
from __future__ import annotations

import daft
from daft import col
from daft.functions import when

from . import config


def add_rule_columns(df: daft.DataFrame) -> daft.DataFrame:
    ok = col("status") == "ok"
    has_face = (
        (col("face_count") > 0)
        & (col("face_score") >= config.FACE_DET_SCORE_MIN)
        & (col("face_area_ratio") >= config.MIN_FACE_RATIO)
    )
    # fill_null(False)：分数为 null 意味着信号缺失（如没检测到脸时
    # face_blur_score 为 null），信号缺失应视为"否"，而不是让 null 向下传播。
    # 最外层的 when(ok, ...) 不带 otherwise，失败行整体落回 null。
    df = df.with_column("has_face", when(ok, has_face.fill_null(False)))
    df = df.with_column(
        "is_blurry",
        when(ok, (col("blur_score") < config.BLUR_THRESHOLD).fill_null(False)),
    )
    df = df.with_column(
        "is_face_blurry",
        when(
            ok,
            (col("has_face") & (col("face_blur_score") < config.FACE_BLUR_THRESHOLD)).fill_null(False),
        ),
    )
    return df
