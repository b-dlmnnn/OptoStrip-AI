import time
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(page_title="OptoStrip-AI | Single-Image Optical Calibrator", page_icon="🧬", layout="wide")


def generate_8tube_strip(ambient_tint="Normal White LED", ntc_contaminated=False, pc_failed=False, seed=42):
    rng = np.random.default_rng(seed)
    h, w = 360, 820
    img = np.full((h, w, 3), (238, 241, 245), dtype=np.uint8)

    # Left White Reference Card (x: 25..105, y: 70..290)
    cv2.rectangle(img, (25, 70), (105, 290), (242, 242, 242), -1)
    cv2.rectangle(img, (25, 70), (105, 290), (160, 165, 175), 2)
    cv2.putText(img, "REF", (42, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 95, 105), 2)

    tube_specs = (
        ("T1", "PC (Pos Ctrl)", not pc_failed),
        ("T2", "NTC (Neg Ctrl)", ntc_contaminated),
        ("T3", "Sample 1", True),
        ("T4", "Sample 2", False),
        ("T5", "Sample 3", True),
        ("T6", "Sample 4", False),
        ("T7", "Sample 5", False),
        ("T8", "Sample 6", True),
    )

    rois = list()
    start_x, step_x, tube_w = 145, 82, 54
    y_top, y_liq, y_bot = 65, 145, 295

    for idx, (code, role, is_pos) in enumerate(tube_specs):
        x1 = start_x + idx * step_x
        x2 = x1 + tube_w
        cx = (x1 + x2) // 2
        rad = tube_w // 2

        if is_pos:
            bgr = (35 + int(rng.integers(-8, 8)), 215 + int(rng.integers(-10, 10)), 245)
        else:
            bgr = (120 + int(rng.integers(-10, 10)), 60 + int(rng.integers(-8, 8)), 225)

        cv2.rectangle(img, (x1 + 3, y_liq), (x2 - 3, y_bot - rad), bgr, -1)
        cv2.circle(img, (cx, y_bot - rad), rad - 3, bgr, -1)
        cv2.rectangle(img, (x1, y_top), (x2, y_bot - rad), (170, 180, 190), 2)
        cv2.ellipse(img, (cx, y_bot - rad), (rad, rad), 0, 0, 180, (170, 180, 190), 2)
        cv2.rectangle(img, (x1 - 3, y_top - 12), (x2 + 3, y_top + 4), (200, 208, 218), -1)
        cv2.rectangle(img, (x1 + 8, y_liq + 8), (x1 + 13, y_bot - rad - 6), (255, 255, 255), -1)

        rois.append({
            "code": code,
            "role": role,
            "box": (x1 + 8, y_liq + 12, x2 - 8, y_bot - rad - 4)
        })

    img_f = img.astype(np.float32)
    if ambient_tint == "Warm Indoor Tungsten (Yellowish)":
        img_f[:, :, 0] *= 0.72
        img_f[:, :, 1] *= 0.96
        img_f[:, :, 2] *= 1.12
    elif ambient_tint == "Cool Shade / Fluorescent (Bluish)":
        img_f[:, :, 0] *= 1.15
        img_f[:, :, 1] *= 0.98
        img_f[:, :, 2] *= 0.76
    elif ambient_tint == "Dim Field Lighting (-35% Brightness)":
        img_f *= 0.65

    noise = rng.normal(0, 4.0, img_f.shape)
    distorted_bgr = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    return distorted_bgr, rois


def auto_detect_and_rectify_strip(bgr_img: np.ndarray):
    resized = cv2.resize(bgr_img, (820, 360), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectified = resized.copy()
    card_found = False
    tilt_angle = 0.0

    # Look for rectangular reference card on the left side (x < 220)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if 3500 < area < 45000:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if cx < 220 and max(rw, rh) > 100:
                card_found = True
                # Normalize tilt angle
                tilt_angle = angle if rw < rh else angle + 90.0
                if abs(tilt_angle) > 0.5 and abs(tilt_angle) < 25.0:
                    M = cv2.getRotationMatrix2D((cx, cy), tilt_angle, 1.0)
                    rectified = cv2.warpAffine(resized, M, (820, 360), borderMode=cv2.BORDER_REPLICATE)
                break

    rois = list()
    start_x, step_x, tube_w = 145, 82, 54
    for idx in range(8):
        x1 = start_x + idx * step_x
        x2 = x1 + tube_w
        role = "PC (Pos Ctrl)" if idx == 0 else ("NTC (Neg Ctrl)" if idx == 1 else f"Sample {idx - 1}")
        rois.append({
            "code": f"T{idx + 1}",
            "role": role,
            "box": (x1 + 8, 155, x2 - 8, 265)
        })
    return rectified, rois, card_found, round(tilt_angle, 2)


def compute_lab_delta_e(bgr_img: np.ndarray, ref_box=(35, 85, 95, 275)):
    rx1, ry1, rx2, ry2 = ref_box
    patch = bgr_img[ry1:ry2, rx1:rx2]
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_L = float(np.mean(lab_patch[:, :, 0])) * (100.0 / 255.0)
    mean_a = float(np.mean(lab_patch[:, :, 1])) - 128.0
    mean_b = float(np.mean(lab_patch[:, :, 2])) - 128.0
    # Ideal neutral white target in CIELAB: L*=95, a*=0, b*=0
    delta_e = float(np.sqrt((mean_L - 95.0) ** 2 + (mean_a - 0.0) ** 2 + (mean_b - 0.0) ** 2))
    return round(delta_e, 2)

def check_image_quality_gate(bgr_img: np.ndarray, ref_box=(35, 85, 95, 275)):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    rx1, ry1, rx2, ry2 = ref_box
    ref_patch = bgr_img[ry1:ry2, rx1:rx2]
    ref_mean = float(np.mean(ref_patch))
    clipped_pct = float(np.mean(np.max(ref_patch, axis=2) >= 252) * 100.0)

    issues = list()
    if blur_score < 45.0:
        issues.append(f"Motion Blur Detected (Sharpness: {blur_score:.1f} < 45.0)")
    if clipped_pct > 15.0:
        issues.append(f"Reference Card Overexposed ({clipped_pct:.1f}% clipped pixels)")
    if ref_mean < 65.0:
        issues.append(f"Insufficient Lighting (Ref Brightness: {ref_mean:.1f} < 65.0)")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "blur_score": round(blur_score, 1),
        "clipped_pct": round(clipped_pct, 1),
        "ref_mean": round(ref_mean, 1),
    }
def calibrate_white_balance(bgr_img: np.ndarray, ref_box=(35, 85, 95, 275)):
    de_before = compute_lab_delta_e(bgr_img, ref_box)
    rx1, ry1, rx2, ry2 = ref_box
    patch = bgr_img[ry1:ry2, rx1:rx2].astype(np.float32)
    mean_bgr = np.mean(patch, axis=(0, 1))
    target_val = 240.0
    gains = np.clip(target_val / np.maximum(mean_bgr, 1.0), 0.5, 2.2)

    calibrated_bgr = np.clip(bgr_img.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    de_after = compute_lab_delta_e(calibrated_bgr, ref_box)
    return calibrated_bgr, gains, de_before, de_after


def analyze_strip_tubes(bgr_img: np.ndarray, rois: list):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    results = list()

    for item in rois:
        x1, y1, x2, y2 = item.get("box")
        roi_lab = lab[y1:y2, x1:x2]
        roi_hsv = hsv[y1:y2, x1:x2]

        valid = (roi_hsv[:, :, 2] <= 225) | (roi_hsv[:, :, 1] >= 40)
        if np.sum(valid) < 10:
            valid = np.ones(roi_lab.shape[:2], dtype=bool)

        b_vals = roi_lab[:, :, 2][valid].astype(np.float32) - 128.0
        a_vals = roi_lab[:, :, 1][valid].astype(np.float32) - 128.0
        mean_b = float(np.mean(b_vals))
        mean_a = float(np.mean(a_vals))

        prob_pos = float(1.0 / (1.0 + np.exp(-np.clip((mean_b - 25.0) / 8.0, -20, 20))))
        if 20.0 <= mean_b <= 30.0:
            status = "INCONCLUSIVE"
        elif mean_b > 30.0:
            status = "POSITIVE"
        else:
            status = "NEGATIVE"

        results.append({
            "Tube": item.get("code"),
            "Role": item.get("role"),
            "CIELAB b*": round(mean_b, 2),
            "CIELAB a*": round(mean_a, 2),
            "P(Positive)": round(prob_pos, 3),
            "Result": status,
            "box": item.get("box")
        })
    return pd.DataFrame(results)
st.sidebar.header("🎛️ OptoStrip-AI Controls")
input_mode = st.sidebar.radio(
    "Image Input Source",
    ("Rule-Based Synthetic Strip", "Upload Real Strip Photo (JPG/PNG)")
)

uploaded_file = None
ambient_mode = "Normal White LED"
sim_ntc_contam = False
sim_pc_fail = False
strip_seed = 10

if input_mode == "Rule-Based Synthetic Strip":
    ambient_mode = st.sidebar.selectbox(
        "Ambient Lighting Condition",
        (
            "Normal White LED",
            "Warm Indoor Tungsten (Yellowish)",
            "Cool Shade / Fluorescent (Bluish)",
            "Dim Field Lighting (-35% Brightness)"
        )
    )
    st.sidebar.subheader("🧪 Batch Quality Control (QC)")
    sim_ntc_contam = st.sidebar.checkbox("Simulate NTC Contamination (T2 Positive)", value=False)
    sim_pc_fail = st.sidebar.checkbox("Simulate Pos. Control Failure (T1 Negative)", value=False)
    strip_seed = st.sidebar.slider("Sample Variation Seed", 1, 50, 10)
else:
    uploaded_file = st.sidebar.file_uploader("Upload 8-Tube Photo with Left White Ref Card", type=("jpg", "jpeg", "png"))
    st.sidebar.caption("Tip: Place a white paper reference card on the left side and 8 colored liquid tubes/cups aligned horizontally.")

st.sidebar.markdown("---")
st.sidebar.info("**Academic Transparency Note:** Single-image web prototype for reference-card white-balance calibration and colorimetric assay readout. No biological nucleic acids were used.")

st.title("🧬 OptoStrip-AI: Single-Image Assay Reader & White-Balance Calibrator")
st.markdown("**Web-Based Reference-Card Color Calibration (ΔE76) & Multi-Tube Batch Quality Control**")

card_detected = True
detected_tilt = 0.0

if input_mode == "Upload Real Strip Photo (JPG/PNG)" and uploaded_file is not None:
    pil_p = Image.open(uploaded_file).convert("RGB")
    bgr_in = cv2.cvtColor(np.array(pil_p), cv2.COLOR_RGB2BGR)
    raw_bgr, rois, card_detected, detected_tilt = auto_detect_and_rectify_strip(bgr_in)
    source_caption = f"Uploaded Photo: {uploaded_file.name} (Tilt Corrected: {detected_tilt}°)"
    if not card_detected:
        st.info("ℹ️ **Auto-Localization Note:** Left white reference card contour was not isolated automatically; using standard grid alignment.")
else:
    raw_bgr, rois = generate_8tube_strip(
        ambient_tint=ambient_mode,
        ntc_contaminated=sim_ntc_contam,
        pc_failed=sim_pc_fail,
        seed=strip_seed
    )
    source_caption = f"Synthetic Strip ({ambient_mode})"

t_start = time.perf_counter()
cal_bgr, rgb_gains, de_before, de_after = calibrate_white_balance(raw_bgr)
qc_gate = check_image_quality_gate(raw_bgr)
if not qc_gate.get("passed"):
    st.error("🛑 **PRE-FLIGHT QUALITY GATE FAILED:** " + " | ".join(qc_gate.get("issues")) + " — Please retake photo under diffuse lighting.")
df_uncal = analyze_strip_tubes(raw_bgr, rois)
df_res = analyze_strip_tubes(cal_bgr, rois)
proc_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
st.caption(f"⏱️ Single-Image Processing Latency: **{proc_ms} ms/image** (CPU Web Runtime)")

# Batch QC Banner
t1_status = df_res.loc[df_res["Tube"] == "T1", "Result"].values.item(0)
t2_status = df_res.loc[df_res["Tube"] == "T2", "Result"].values.item(0)

if t2_status == "POSITIVE":
    st.error("🚨 **BATCH INVALID (NTC CONTAMINATION):** Tube T2 (Negative Control) amplified! Discard batch and re-run assay.")
elif t1_status == "NEGATIVE":
    st.warning("⚠️ **BATCH INVALID (POSITIVE CONTROL FAILED):** Tube T1 (Positive Control) showed no signal. Check reagents/lighting.")
else:
    pos_samples = df_res[(df_res["Tube"].isin(("T3", "T4", "T5", "T6", "T7", "T8"))) & (df_res["Result"] == "POSITIVE")]
    st.success(f"✅ **VALID BATCH RUN:** Controls Passed (T1=POS, T2=NEG) | **{len(pos_samples)} / 6** Field Samples Detected Positive.")

# Delta E & White Balance Telemetry
m1, m2, m3, m4 = st.columns(4)
de_reduction = round(de_before - de_after, 2)
m1.metric("White Card ΔE76 (Pre → Post)", f"{de_before} → {de_after}", delta=f"-{de_reduction} ΔE76", delta_color="inverse")
m2.metric("Blue Gain Factor", f"{rgb_gains.item(0):.2f}x")
m3.metric("Green Gain Factor", f"{rgb_gains.item(1):.2f}x")
m4.metric("Red Gain Factor", f"{rgb_gains.item(2):.2f}x")

c_before, c_after = st.columns(2)
with c_before:
    st.subheader("1. Raw Camera Capture (Uncalibrated)")
    st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption=source_caption, use_container_width=True)

with c_after:
    st.subheader("2. White-Balance Calibrated & Segmented")
    overlay = cal_bgr.copy()
    cv2.rectangle(overlay, (35, 85), (95, 275), (0, 140, 255), 3)
    for _, row in df_res.iterrows():
        x1, y1, x2, y2 = row["box"]
        color = (0, 0, 230) if row["Result"] == "POSITIVE" else ((0, 165, 255) if row["Result"] == "INCONCLUSIVE" else (0, 175, 60))
        cv2.rectangle(overlay, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 2)
        cv2.putText(overlay, row["Tube"], (x1 + 4, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Calibrated Output (Orange=Ref ROI, Red=POS, Green=NEG)", use_container_width=True)

st.markdown("---")
col_tbl, col_cht = st.columns((1.25, 1.0))

with col_tbl:
    st.subheader("3. Uncalibrated vs. Calibrated Baseline Comparison")
    comp_df = pd.DataFrame({
        "Tube": df_res["Tube"],
        "Role": df_res["Role"],
        "Raw b* (Uncalib.)": df_uncal["CIELAB b*"],
        "Raw Decision": df_uncal["Result"],
        "Calib. b*": df_res["CIELAB b*"],
        "Calib. Decision": df_res["Result"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download ΔE & Batch Comparison Report (CSV)", data=csv_bytes, file_name="optostrip_de_report.csv", mime="text/csv")

with col_cht:
    st.subheader("4. Signal Shift: Uncalibrated vs. Calibrated (CIELAB b*)")
    plot_df = pd.DataFrame({
        "Tube": list(df_res["Tube"]) * 2,
        "CIELAB b*": list(df_uncal["CIELAB b*"]) + list(df_res["CIELAB b*"]),
        "Stage": ["1. Uncalibrated (Raw)"] * 8 + ["2. Calibrated (WB Fixed)"] * 8
    })
    fig_bar = px.bar(
        plot_df,
        x="Tube",
        y="CIELAB b*",
        color="Stage",
        barmode="group",
        color_discrete_map={"1. Uncalibrated (Raw)": "#94a3b8", "2. Calibrated (WB Fixed)": "#2563eb"}
    )
    fig_bar.add_hline(y=25.0, line_dash="dash", line_color="#dc2626", annotation_text="Threshold (b*=25)")
    fig_bar.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("---")
st.header("📊 Multi-Lighting Benchmark & Failure Modes Analysis")
st.caption("Statistical evaluation across 80 synthetic strip trials (20 trials × 4 lighting conditions) comparing Uncalibrated vs. Reference-Calibrated performance.")


@st.cache_data
def run_lighting_benchmark():
    conditions = (
        "Normal White LED",
        "Warm Indoor Tungsten (Yellowish)",
        "Cool Shade / Fluorescent (Bluish)",
        "Dim Field Lighting (-35% Brightness)"
    )
    ground_truth = ("POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE")
    bench_rows = list()
    summary_rows = list()

    for cond in conditions:
        uncal_correct = 0
        cal_correct = 0
        total_tubes = 0

        for trial in range(20):
            s_bgr, s_rois = generate_8tube_strip(ambient_tint=cond, seed=500 + trial)
            c_bgr, _, de_pre, de_post = calibrate_white_balance(s_bgr)

            bench_rows.append({"Lighting Condition": cond, "Stage": "1. Uncalibrated (Raw)", "White Card ΔE": de_pre})
            bench_rows.append({"Lighting Condition": cond, "Stage": "2. Calibrated (WB Fixed)", "White Card ΔE": de_post})

            df_u = analyze_strip_tubes(s_bgr, s_rois)
            df_c = analyze_strip_tubes(c_bgr, s_rois)

            for idx, gt in enumerate(ground_truth):
                if df_u["Result"].values.item(idx) == gt:
                    uncal_correct += 1
                if df_c["Result"].values.item(idx) == gt:
                    cal_correct += 1
                total_tubes += 1

        summary_rows.append({
            "Lighting Condition": cond,
            "Uncalibrated Accuracy (%)": round((uncal_correct / total_tubes) * 100.0, 1),
            "Calibrated Accuracy (%)": round((cal_correct / total_tubes) * 100.0, 1),
            "Accuracy Gain (%)": f"+{round(((cal_correct - uncal_correct) / total_tubes) * 100.0, 1)}%"
        })

    return pd.DataFrame(bench_rows), pd.DataFrame(summary_rows)


bench_df, summary_df = run_lighting_benchmark()

b_col1, b_col2 = st.columns((1.15, 1.0))
with b_col1:
    st.subheader("White-Balance ΔE Distribution by Lighting")
    fig_box = px.box(
        bench_df,
        x="Lighting Condition",
        y="White Card ΔE",
        color="Stage",
        color_discrete_map={"1. Uncalibrated (Raw)": "#94a3b8", "2. Calibrated (WB Fixed)": "#2563eb"}
    )
    fig_box.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_box, use_container_width=True)

with b_col2:
    st.subheader("Classification Accuracy: Raw vs. Calibrated")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.info(
        "**Known Limitations & Failure Modes:**\n"
        "- **Specular Clipping:** Direct flash glare (>252 AU) saturates RGB channels and cannot be recovered mathematically.\n"
        "- **Extreme Tilt (>25°):** Steep camera angles distort tube ROI alignment.\n"
        "- **Single Assay Format:** Currently calibrated for Phenol Red (yellow/pink) colorimetric chemistry."
    )
