# ==============================
# 🚀 HEMVI PARKING DETECTOR  v5.0
# ==============================
# Fix vs v4:
#   🌳 Vegetation mask — trees/grass detected via HSV green range
#      and fully excluded from parking slots
#   🔧 Slot strips now avoid ANY column that has tree pixels
#   🔧 Added tree sensitivity slider (tune if over/under-masking)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from google.colab import files
from IPython.display import display, HTML
import ipywidgets as widgets

# ─────────────────────────────────────────────────────────────────
# STEP 1  UPLOAD + AUTO-ORIENT
# ─────────────────────────────────────────────────────────────────
print("📸  Upload your top-down / satellite street image...")
uploaded   = files.upload()
image_path = list(uploaded.keys())[0]
img        = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Cannot read '{image_path}'.")

h, w = img.shape[:2]
if h > w:
    img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    print(f"  ↩️  Auto-rotated → {w}×{h} px")
else:
    print(f"  ✅  {w}×{h} px")


# ─────────────────────────────────────────────────────────────────
# STEP 2  SETTINGS UI
# ─────────────────────────────────────────────────────────────────
CAR_PRESETS = {
    "Compact   (3.8 m)": 3.8,
    "Sedan     (4.5 m)": 4.5,
    "SUV / MPV (4.8 m)": 4.8,
    "Large SUV (5.1 m)": 5.1,
    "Van       (5.5 m)": 5.5,
    "Custom":             None,
}
W = widgets.Layout

preset_dd   = widgets.Dropdown(options=list(CAR_PRESETS.keys()),
                                value="Sedan     (4.5 m)",
                                description="Car type:",
                                style={"description_width":"initial"},
                                layout=W(width="250px"))
custom_len  = widgets.FloatText(value=4.5, description="Length (m):",
                                style={"description_width":"initial"},
                                layout=W(width="180px"), disabled=True)
car_wid_in  = widgets.FloatText(value=2.2, min=1.0, max=5.0, step=0.1,
                                description="Width (m):",
                                style={"description_width":"initial"},
                                layout=W(width="180px"))
ppm_in      = widgets.FloatText(value=21.0, description="PPM:",
                                style={"description_width":"initial"},
                                layout=W(width="140px"))
buf_in      = widgets.FloatText(value=0.5, step=0.1, description="Buffer (m):",
                                style={"description_width":"initial"},
                                layout=W(width="180px"))
road_top_sl = widgets.IntSlider(value=20, min=0, max=100, step=1,
                                description="Road band top %:",
                                style={"description_width":"initial"},
                                layout=W(width="420px"))
road_bot_sl = widgets.IntSlider(value=80, min=0, max=100, step=1,
                                description="Road band bot %:",
                                style={"description_width":"initial"},
                                layout=W(width="420px"))
road_thr_sl = widgets.IntSlider(value=85, min=30, max=180, step=5,
                                description="Road darkness:",
                                style={"description_width":"initial"},
                                layout=W(width="420px"))
# NEW: vegetation sensitivity
veg_sat_sl  = widgets.IntSlider(value=40, min=10, max=120, step=5,
                                description="Tree sensitivity (sat):",
                                style={"description_width":"initial"},
                                layout=W(width="420px"))
debug_chk   = widgets.Checkbox(value=False, description="Show debug masks",
                                layout=W(width="200px"))
run_btn     = widgets.Button(description="🔍  Detect Parking Spaces",
                              button_style="success",
                              layout=W(width="260px", height="38px"))
status_lbl  = widgets.Label(value="")

def _on_preset(change):
    custom_len.disabled = (change["new"] != "Custom")
preset_dd.observe(_on_preset, names="value")

display(HTML("<h3 style='font-family:sans-serif;margin-bottom:4px'>⚙️  Configure Detection</h3>"))
display(widgets.VBox([
    widgets.HBox([preset_dd, custom_len]),
    widgets.HBox([car_wid_in, ppm_in, buf_in]),
    widgets.HTML("<b style='font-family:sans-serif;font-size:13px'>Road band</b>"),
    road_top_sl, road_bot_sl, road_thr_sl,
    widgets.HTML("<b style='font-family:sans-serif;font-size:13px'>Vegetation exclusion</b>"),
    veg_sat_sl,
    widgets.HTML(
        "<small style='color:#666;font-family:sans-serif'>"
        "<b>Tree sensitivity</b>: raise this value if sandy terrain is being "
        "treated as trees. Lower it if trees are not fully masked.</small>"
    ),
    debug_chk,
    widgets.HBox([run_btn, status_lbl]),
]))


# ─────────────────────────────────────────────────────────────────
# STEP 3  VEGETATION MASK  (NEW)
# ─────────────────────────────────────────────────────────────────
def _make_veg_mask(img: np.ndarray, min_saturation: int) -> np.ndarray:
    """
    Detects trees, grass and shrubs using HSV colour space.

    Trees from above have:
      • Hue  35–85   (yellow-green → green → teal-green)
      • Saturation > min_saturation  (distinguishes green from grey road / sandy terrain)
      • Value > 30   (not too dark — avoids black shadows)

    The mask is dilated to cover the full canopy, not just the brightest
    pixels, so slot gaps never accidentally land inside a tree crown.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([35, min_saturation, 30],  dtype=np.uint8)
    upper = np.array([85, 255,            255],  dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Expand outward to capture full canopy silhouette
    kernel = np.ones((12, 12), np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=2)
    return mask


# ─────────────────────────────────────────────────────────────────
# STEP 4  ROAD MASK
# ─────────────────────────────────────────────────────────────────
def _make_road_mask(img, road_y1, road_y2, dark_thresh):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    full  = np.zeros(gray.shape, dtype=np.uint8)
    region = gray[road_y1:road_y2].copy()
    _, dark = cv2.threshold(region, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  np.ones((10, 10), np.uint8))
    full[road_y1:road_y2] = dark
    return full


def _find_road_edges(road_mask, road_y1, road_y2):
    band     = road_mask[road_y1:road_y2]
    row_frac = (band > 0).mean(axis=1)
    rows     = np.where(row_frac > 0.25)[0]
    if len(rows) == 0:
        mid = (road_y2 - road_y1) // 2
        return road_y1 + mid // 4, road_y1 + 3 * mid // 4
    return road_y1 + int(rows[0]), road_y1 + int(rows[-1])


# ─────────────────────────────────────────────────────────────────
# STEP 5  VEHICLE DETECTOR
# ─────────────────────────────────────────────────────────────────
def _detect_boxes(img, road_y1, road_y2):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (5, 5), 0)

    mask_a = cv2.adaptiveThreshold(blur, 255,
                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -6)
    edges  = cv2.Canny(blur, 25, 80)
    mask_b = cv2.dilate(edges, np.ones((4, 4), np.uint8), iterations=2)
    _, mask_c = cv2.threshold(hsv[:, :, 1], 30, 255, cv2.THRESH_BINARY)
    grad   = cv2.morphologyEx(eq, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, mask_d = cv2.threshold(grad, 18, 255, cv2.THRESH_BINARY)
    mask_d = cv2.dilate(mask_d, np.ones((4, 4), np.uint8), iterations=1)

    fused = cv2.bitwise_or(mask_a, mask_b)
    fused = cv2.bitwise_or(fused,  mask_c)
    fused = cv2.bitwise_or(fused,  mask_d)
    solid = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (60 < area < 8_000):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if not (road_y1 <= (y + bh / 2) <= road_y2):
            continue
        if max(bw, bh) / (min(bw, bh) + 1e-6) > 7.0:
            continue
        hull_a = cv2.contourArea(cv2.convexHull(cnt))
        if area / (hull_a + 1e-6) < 0.30:
            continue
        boxes.append([x, y, x + bw, y + bh])

    return boxes, solid


# ─────────────────────────────────────────────────────────────────
# STEP 6  SLOT FINDER — now blocks trees in addition to road
# ─────────────────────────────────────────────────────────────────
def _find_slots(boxes, strip_y1, strip_y2,
                road_mask, veg_mask,          # ← NEW: veg_mask passed in
                img_w, car_len_px, buffer_px):
    """
    A column is blocked if it contains:
      • A detected vehicle (+ buffer)
      • Road surface pixels
      • Vegetation / tree pixels  ← NEW
    """
    if strip_y2 <= strip_y1:
        return []

    occ = np.zeros(img_w, dtype=np.uint8)

    # Block vehicles
    for b in boxes:
        mid_y = (b[1] + b[3]) / 2
        if strip_y1 <= mid_y <= strip_y2:
            occ[max(0, b[0] - buffer_px) : min(img_w, b[2] + buffer_px)] = 1

    # Block road pixels within the strip
    if road_mask is not None:
        road_cols = (road_mask[strip_y1:strip_y2] > 0).any(axis=0)
        occ[road_cols] = 1

    # Block tree / vegetation pixels within the strip  ← KEY FIX
    if veg_mask is not None:
        veg_cols = (veg_mask[strip_y1:strip_y2] > 0).any(axis=0)
        occ[veg_cols] = 1

    # Sweep free runs → chop into car_len_px slots
    slots      = []
    free_start = None

    for x in range(img_w):
        if occ[x] == 0 and free_start is None:
            free_start = x
        elif (occ[x] == 1 or x == img_w - 1) and free_start is not None:
            run_end = x if occ[x] == 1 else x + 1
            n       = (run_end - free_start) // car_len_px
            for i in range(n):
                x1 = free_start + i * car_len_px
                slots.append({"x1": x1, "y1": strip_y1,
                               "x2": x1 + car_len_px, "y2": strip_y2})
            free_start = None

    return slots


# ─────────────────────────────────────────────────────────────────
# STEP 7  MAIN RUN
# ─────────────────────────────────────────────────────────────────
def _run(btn):
    status_lbl.value = "  ⏳ running…"

    PPM        = ppm_in.value
    BUFFER_PX  = int(buf_in.value * PPM)
    car_len_m  = (custom_len.value if preset_dd.value == "Custom"
                  else CAR_PRESETS[preset_dd.value])
    car_wid_m  = car_wid_in.value
    CAR_LEN_PX = max(1, int(car_len_m * PPM))
    CAR_WID_PX = max(1, int(car_wid_m * PPM))

    band_y1 = int(road_top_sl.value / 100 * h)
    band_y2 = int(road_bot_sl.value / 100 * h)
    dark_th  = road_thr_sl.value
    min_sat  = veg_sat_sl.value

    # Masks
    road_mask = _make_road_mask(img, band_y1, band_y2, dark_th)
    veg_mask  = _make_veg_mask(img, min_sat)           # ← NEW
    road_top, road_bot = _find_road_edges(road_mask, band_y1, band_y2)

    # Parking strips (one car-width above/below road edges)
    top_y1 = max(0,     road_top - CAR_WID_PX)
    top_y2 = road_top
    bot_y1 = road_bot
    bot_y2 = min(h - 1, road_bot + CAR_WID_PX)

    # Vehicles
    boxes, dbg_vehicle = _detect_boxes(img, band_y1, band_y2)

    # Slots
    top_slots = _find_slots(boxes, top_y1, top_y2,
                             road_mask, veg_mask, w, CAR_LEN_PX, BUFFER_PX)
    bot_slots = _find_slots(boxes, bot_y1, bot_y2,
                             road_mask, veg_mask, w, CAR_LEN_PX, BUFFER_PX)
    all_slots = top_slots + bot_slots

    status_lbl.value = (f"  ✅  {len(boxes)} vehicles · "
                        f"{len(top_slots)} top · {len(bot_slots)} bottom slots")

    # ── Render ──────────────────────────────────────────────────────
    out     = img.copy()
    overlay = img.copy()

    # Road tint (red)
    road_vis = np.zeros_like(img)
    road_vis[road_mask > 0] = (0, 0, 180)
    cv2.addWeighted(road_vis, 0.25, out, 0.75, 0, out)

    # Tree tint (dark green)  ← NEW visual feedback
    veg_vis = np.zeros_like(img)
    veg_vis[veg_mask > 0] = (0, 120, 0)
    cv2.addWeighted(veg_vis, 0.30, out, 0.70, 0, out)

    # Parking strip outlines
    cv2.rectangle(out, (0, top_y1), (w - 1, top_y2), (0, 220, 220), 1)
    cv2.rectangle(out, (0, bot_y1), (w - 1, bot_y2), (0, 220, 220), 1)

    # Detected vehicles
    for b in boxes:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (210, 30, 30), 2)

    # Parking slots — green filled boxes the size of the car
    for s in all_slots:
        cv2.rectangle(overlay, (s["x1"], s["y1"]), (s["x2"], s["y2"]),
                      (30, 210, 60), -1)
        cv2.rectangle(out,     (s["x1"], s["y1"]), (s["x2"], s["y2"]),
                      (10, 160, 40), 1)
        cx = (s["x1"] + s["x2"]) // 2
        cy = (s["y1"] + s["y2"]) // 2
        fs = max(0.25, min(0.5, CAR_LEN_PX / 80))
        cv2.putText(out, "P", (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),     2, cv2.LINE_AA)
        cv2.putText(out, "P", (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 255, 100), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.30, out, 0.70, 0, out)

    # ── Plot ────────────────────────────────────────────────────────
    n_rows = 3 if debug_chk.value else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(22, 5 * n_rows), squeeze=False)

    ax = axes[0][0]
    ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(
        f"HEMVI v5  |  Car {car_len_m}m × {car_wid_m}m  "
        f"|  {PPM} px/m  |  Buffer {buf_in.value}m",
        fontsize=11, pad=8,
    )
    ax.legend(handles=[
        mpatches.Patch(color="#1e1eb4", label=f"Detected vehicles ({len(boxes)})"),
        mpatches.Patch(color="#1ed23c", label=f"Parking slots ({len(all_slots)}) — no road, no trees"),
        mpatches.Patch(color="#b40000", alpha=0.4, label="Road surface (excluded)"),
        mpatches.Patch(color="#007800", alpha=0.5, label="Vegetation (excluded)"),
        mpatches.Patch(color="#00c8c8", label="Parking strips"),
    ], loc="lower right", fontsize=9, framealpha=0.85)

    if debug_chk.value:
        axes[1][0].imshow(road_mask, cmap="Reds")
        axes[1][0].set_title("Road mask", fontsize=10)
        axes[1][0].axis("off")

        # Show vegetation mask in green
        veg_rgb = np.zeros((*veg_mask.shape, 3), dtype=np.uint8)
        veg_rgb[veg_mask > 0] = [0, 200, 0]
        axes[2][0].imshow(veg_rgb)
        axes[2][0].set_title(
            f"Vegetation mask  (sat threshold = {min_sat})  "
            "← raise slider if sandy terrain is caught here", fontsize=10)
        axes[2][0].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"\n{'─'*52}")
    print(f"  Vehicles detected  : {len(boxes)}")
    print(f"  Road edges         : top y={road_top}  bot y={road_bot}")
    print(f"  Parking strips     : top [{top_y1}–{top_y2}]  bot [{bot_y1}–{bot_y2}]")
    print(f"  Slot size          : {CAR_LEN_PX}px × {CAR_WID_PX}px  ({car_len_m}m × {car_wid_m}m)")
    print(f"  Top slots          : {len(top_slots)}")
    print(f"  Bottom slots       : {len(bot_slots)}")
    print(f"  Total slots        : {len(all_slots)}")
    print(f"{'─'*52}\n")


run_btn.on_click(_run)