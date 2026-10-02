"""
工件表面缺陷检测 —— 最小可运行闭环（第一天 demo）

这个脚本做一件工业视觉里最典型的事：
    输入一张工件图 -> 找出表面缺陷 -> 画出位置并给出数量

它刻意不依赖任何外部数据集：工件图由程序自己合成，
所以你 clone 下来就能跑，不用先折腾下载和路径。

跑法：
    python detect_defects.py

产出：
    output/result.png   原图 / 缺陷掩膜 / 检测结果 三合一对比图
    output/mask.png     单独的缺陷掩膜
"""

import os
import cv2
import numpy as np

# ---------- 0. 输出目录 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 1. 合成一张模拟工件图 ----------
def make_sample_workpiece(w=520, h=380, seed=7):
    """
    画一张"金属工件"图：
      - 深色背景 + 圆形工件本体
      - 一处凹坑（比周围暗）
      - 一条划痕（比周围亮）
      - 加上光照不均和传感器噪声，让它更接近真实相机拍出来的样子

    真实项目里这一步会换成 cv2.imread("你的图片.jpg")。
    """
    rng = np.random.default_rng(seed)

    img = np.full((h, w), 28, np.uint8)                 # 背景
    cv2.circle(img, (w // 2, h // 2), 140, 190, -1)     # 工件本体
    cv2.circle(img, (w // 2 - 45, h // 2 - 30), 17, 92, -1)    # 缺陷 1：凹坑
    cv2.line(img, (w // 2 + 5, h // 2 + 48),
             (w // 2 + 95, h // 2 + 8), 236, 4)          # 缺陷 2：划痕

    # 光照不均：左侧亮、右侧暗
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.clip(img.astype(np.float32) * (1.0 - 0.28 * xx / w), 0, 255)

    # 传感器噪声
    img = img + rng.normal(0, 5.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- 2. 核心检测逻辑 ----------
def detect_defects(gray, blur_ksize=5, bg_ksize=41, thresh=20,
                   area_min=40, area_max=4000):
    """
    思路（工业上处理"光照不均"的标准做法）：

      先想一个坑：如果直接用"全图灰度均值 ± k 倍标准差"来卡阈值，
      因为相机打光不均（左边亮右边暗），右侧整体偏暗，
      缺陷就容易被整体压低到阈值以下 —— 直接漏检。

      正确做法不是把阈值调得更激进，而是先把"光照本底"估出来减掉：

      1) 用一个大核做形态学开运算，得到的是"平滑的表面本底"，
         原图减去它 —— 剩下的就是比本底亮的细小特征（顶帽变换）
      2) 同样，用闭运算估本底，本底减去原图 ——
         得到比本底暗的细小特征（底帽变换）
      3) 光照不均在这两步里被自动抵消掉了，剩下的差值才是缺陷的真实对比度

      然后：

      4) 定位工件区域，去掉边缘（边缘在形态学里必然产生伪影）
      5) 差值超过 thresh 的像素判为缺陷
      6) 连通域筛选，按面积去掉噪点和假目标

    参数 thresh 就是"多亮/多暗才算缺陷"，是这套算法最关键的旋钮。

    返回 (缺陷框列表, 缺陷掩膜)
    """
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 1) 定位工件：大范围内的阈值分割，取面积最大的轮廓
    _, part_mask = cv2.threshold(blurred, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], np.zeros_like(gray)

    part = max(contours, key=cv2.contourArea)
    part_region = np.zeros_like(gray)
    cv2.drawContours(part_region, [part], -1, 255, -1)
    # 往里收 30 像素：形态学操作在边界处一定产生伪影，先排除掉
    part_region = cv2.erode(part_region, np.ones((61, 61), np.uint8))

    # 2) 估计光照本底，用顶帽 / 底帽把不均匀光照抵消掉
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (bg_ksize, bg_ksize))
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, bg_kernel)
    bright_diff = cv2.subtract(blurred, background)          # 顶帽：亮特征
    dark_diff = cv2.subtract(cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, bg_kernel),
                             blurred)                        # 底帽：暗特征

    # 3) 双向阈值 + 限制在工件内部
    _, bright_mask = cv2.threshold(bright_diff, thresh, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(dark_diff, thresh, 255, cv2.THRESH_BINARY)
    defect_mask = cv2.bitwise_or(bright_mask, dark_mask)
    defect_mask = cv2.bitwise_and(defect_mask, part_region)

    # 4) 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5) 连通域筛选
    num, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, 8)
    boxes = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < area_min or area > area_max:
            continue
        x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        boxes.append((x, y, bw, bh, area))

    boxes.sort(key=lambda b: -b[4])
    return boxes, defect_mask


# ---------- 3. 可视化 ----------
def draw_result(gray, boxes):
    """把灰度图转成 BGR 并画框。OpenCV 的 putText 画不了中文，所以标注用英文。"""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for idx, (x, y, w, h, area) in enumerate(boxes, start=1):
        cv2.rectangle(vis, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 0, 255), 2)
        label = f"D{idx}  {area}px"
        tx, ty = x - 6, max(18, y - 12)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def to_bgr(img2d):
    return cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)


def label_panel(img, text):
    """在每张面板左上角贴一个标题条"""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (40, 40, 40), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------- 4. 主流程 ----------
def main():
    gray = make_sample_workpiece()
    cv2.imwrite(os.path.join(OUT_DIR, "input.png"), gray)

    boxes, mask = detect_defects(gray)

    print("=" * 46)
    print(f"检测到缺陷数量：{len(boxes)}")
    for idx, (x, y, w, h, area) in enumerate(boxes, start=1):
        print(f"  D{idx}  位置=({x:>3},{y:>3})  尺寸={w:>3}x{h:<3}  面积={area:>5}px")
    print("=" * 46)

    result = draw_result(gray, boxes)
    cv2.imwrite(os.path.join(OUT_DIR, "mask.png"), mask)

    panel = np.hstack([label_panel(to_bgr(gray), "input"),
                       label_panel(to_bgr(mask), "defect mask"),
                       label_panel(result, f"result  ({len(boxes)} defects)")])
    out_path = os.path.join(OUT_DIR, "result.png")
    cv2.imwrite(out_path, panel)

    print(f"结果图已保存：{out_path}")


if __name__ == "__main__":
    main()
