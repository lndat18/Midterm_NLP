import os
import re
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Phân loại Bình luận Toxic",
    page_icon="😡",
    layout="centered",
    initial_sidebar_state="auto",
)

# ==================== TIỆN ÍCH LƯU FILE ====================
APP_ROOT = Path(__file__).resolve().parent
LOG_DIR = (APP_ROOT / ".." / "data" / "app_logs").resolve()
TRAIN_APPEND_PATH = (APP_ROOT / ".." / "data" / "processed" / "new_comments_for_training.csv").resolve()
LOG_PATH = (LOG_DIR / "predictions_log.csv").resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_APPEND_PATH.parent.mkdir(parents=True, exist_ok=True)

def append_csv_row(csv_path: Path, row: dict):
    existed = csv_path.exists()
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", index=False, header=not existed, encoding="utf-8-sig")

# ==================== TIỀN XỬ LÝ (GIỐNG LÚC TRAIN) ====================
def preprocess_text(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ==================== TẢI MODEL ====================
MODEL_PATH = (APP_ROOT / ".." / "reports" / "final_pipeline_Decision Tree_Bag-of-Words_20251012_075841.joblib").resolve()
st.caption(f"📦 Model file: `{MODEL_PATH}`")

if not MODEL_PATH.exists():
    st.error(f"Không tìm thấy file model tại: `{MODEL_PATH}`")
    st.stop()

# một số pipeline có bước từ imblearn, cần gói imbalanced-learn để unpickle
try:
    import imblearn  # noqa: F401
except Exception:
    st.error("Thiếu gói **imbalanced-learn**. Cài: `pip install imbalanced-learn`")
    st.stop()

try:
    model_pipeline = joblib.load(str(MODEL_PATH))
except Exception as e:
    st.error(f"Lỗi khi nạp model: {e}")
    st.stop()

# Lấy classes_
classes = getattr(model_pipeline, "classes_", None)
if classes is None:
    try:
        classes = model_pipeline.named_steps["clf"].classes_
    except Exception:
        classes = [0, 1]

# Map hiển thị
try:
    classes_int = [int(c) for c in classes]
    label_display = {classes_int[0]: "Non-toxic", classes_int[1]: "Toxic"}
    proba_cols_default = [label_display.get(int(c), str(c)) for c in classes]
except Exception:
    label_display = {classes[0]: str(classes[0]), classes[1]: str(classes[1])}
    proba_cols_default = [str(c) for c in classes]

# ==================== UI ====================
st.title("😡 Phân loại Bình luận Toxic (Tiếng Việt)")
st.write(
    "• **Single**: nhập 1 bình luận\n"
    "• **Batch**: dán nhiều dòng hoặc tải 1 hay nhiều CSV (có cột `text`). "
    "Bạn có thể **sửa nhãn** theo từng dòng rồi lưu."
)

tab_single, tab_paste, tab_csv = st.tabs(["Một bình luận", "Dán nhiều dòng", "Tải CSV (nhiều file)"])

# ---------- TAB 1: SINGLE ----------
with tab_single:
    # helper reset for single box
    def _reset_single():
        st.session_state["single_text"] = ""

    with st.form("single_form", clear_on_submit=True):  # auto-clear after submit
        user_input = st.text_area(
            "Nhập 1 bình luận:",
            height=140,
            placeholder="Ví dụ: Sản phẩm này thật tuyệt vời!",
            key="single_text",
        )
        submitted_single = st.form_submit_button("Phân loại")

    if submitted_single and user_input.strip():
        with st.spinner("Đang phân tích..."):
            processed_input = preprocess_text(user_input)
            y_pred = model_pipeline.predict([processed_input])[0]
            try:
                proba = model_pipeline.predict_proba([processed_input])[0]
            except Exception:
                proba = None

            try:
                label_id = int(y_pred)
            except Exception:
                label_id = y_pred
            label_name = label_display.get(label_id, str(label_id))

            proba_cols = proba_cols_default
            confidence_score = None
            if proba is not None:
                try:
                    idx = proba_cols.index(label_name)
                    confidence_score = float(proba[idx])
                except Exception:
                    pass

        st.subheader("Kết quả phân loại:")
        if label_name.lower().startswith("non"):
            st.success(f"✅ **{label_name}** (Không độc hại)")
        else:
            st.error(f"😡 **{label_name}** (Độc hại)")
        if confidence_score is not None:
            st.write(f"Độ tin cậy: **{confidence_score:.2%}**")

        if proba is not None:
            st.subheader("Chi tiết xác suất:")
            df_proba = pd.DataFrame([proba], columns=proba_cols, index=["Xác suất"])
            st.dataframe(df_proba.style.format("{:.2%}"), use_container_width=True)

        st.markdown("---")
        st.subheader("💾 Lưu bình luận này")
        c1, c2 = st.columns(2)
        with c1:
            corrected_label = st.selectbox(
                "Gán nhãn muốn lưu:",
                options=["Giữ dự đoán", "Non-toxic", "Toxic"],
                index=0,
                key="single_correct_label",
            )
        with c2:
            also_append_train = st.checkbox("Thêm vào tập huấn luyện mới", value=True, key="single_append")

        note = st.text_input("Ghi chú (tuỳ chọn):", placeholder="Nguồn, bối cảnh, người gán nhãn...", key="single_note")

        if st.button("Lưu bình luận", key="single_save_btn"):
            save_label_name = label_name if corrected_label == "Giữ dự đoán" else corrected_label
            if save_label_name.lower().startswith("non"):
                save_label_id = 0
            elif save_label_name.lower().startswith("toxic"):
                save_label_id = 1
            else:
                try:
                    save_label_id = int(save_label_name)
                except Exception:
                    save_label_id = label_id

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_row = {
                "timestamp": timestamp,
                "raw_text": user_input,
                "processed_text": processed_input,
                "pred_label_name": label_name,
                "pred_label_id": label_id,
                "saved_label_name": save_label_name,
                "saved_label_id": save_label_id,
                "confidence": confidence_score if confidence_score is not None else "",
                "note": note,
            }
            if proba is not None:
                for cname, p in zip(proba_cols, proba):
                    log_row[f"proba::{cname}"] = p
            append_csv_row(LOG_PATH, log_row)

            if also_append_train:
                train_row = {
                    "text_bow_tfidf": processed_input,
                    "predicted_label_id": save_label_id,
                    "source": "app-feedback",
                    "timestamp": timestamp,
                    "note": note,
                }
                append_csv_row(TRAIN_APPEND_PATH, train_row)

            st.success(
                f"Đã lưu vào:\n- Log: `{LOG_PATH}`"
                + (f"\n- Tập huấn luyện mới: `{TRAIN_APPEND_PATH}`" if also_append_train else "")
            )

        st.button("🔄 Nhập bình luận mới", on_click=_reset_single)

# ---------- helpers dùng chung cho batch ----------
def _predict_batch(texts: list[str]):
    proc = [preprocess_text(x) for x in texts]
    y = model_pipeline.predict(proc)
    try:
        P = model_pipeline.predict_proba(proc)
    except Exception:
        P = None

    def to_lbl_id(v):
        try:
            return int(v)
        except Exception:
            return v

    pred_ids = [to_lbl_id(v) for v in y]
    pred_names = [label_display.get(v, str(v)) for v in pred_ids]

    conf = []
    if P is not None:
        proba_cols = proba_cols_default
        for name, row in zip(pred_names, P):
            try:
                idx = proba_cols.index(name)
                conf.append(float(row[idx]))
            except Exception:
                conf.append(None)
    else:
        proba_cols = None
        conf = [None] * len(pred_names)

    df = pd.DataFrame({
        "raw_text": texts,
        "processed_text": proc,
        "pred_label_name": pred_names,
        "pred_label_id": pred_ids,
        "confidence": conf,
    })
    if P is not None:
        for j, cname in enumerate(proba_cols):
            df[f"proba::{cname}"] = P[:, j]

    def _default_saved(nm):
        nm_l = str(nm).lower()
        if nm_l.startswith("non"):
            return "Non-toxic"
        if nm_l.startswith("toxic"):
            return "Toxic"
        return "Giữ dự đoán"

    df["saved_label_name"] = [_default_saved(n) for n in pred_names]
    return df

def _save_batch(df_edited: pd.DataFrame, note: str, append_train: bool):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for _, row in df_edited.iterrows():
        save_label_name = (
            row["pred_label_name"]
            if row["saved_label_name"] == "Giữ dự đoán"
            else row["saved_label_name"]
        )
        # 0/1 fallback
        if str(save_label_name).lower().startswith("non"):
            save_label_id = 0
        elif str(save_label_name).lower().startswith("toxic"):
            save_label_id = 1
        else:
            try:
                save_label_id = int(save_label_name)
            except Exception:
                save_label_id = row["pred_label_id"]

        log_row = {
            "timestamp": timestamp,
            "raw_text": row["raw_text"],
            "processed_text": row["processed_text"],
            "pred_label_name": row["pred_label_name"],
            "pred_label_id": row["pred_label_id"],
            "saved_label_name": save_label_name,
            "saved_label_id": save_label_id,
            "confidence": row.get("confidence", ""),
            "note": note,
        }
        for c in df_edited.columns:
            if c.startswith("proba::"):
                log_row[c] = row.get(c, "")
        append_csv_row(LOG_PATH, log_row)

        if append_train:
            train_row = {
                "text_bow_tfidf": row["processed_text"],
                "predicted_label_id": save_label_id,
                "source": "app-feedback-batch",
                "timestamp": timestamp,
                "note": note,
            }
            append_csv_row(TRAIN_APPEND_PATH, train_row)
        n += 1
    return n

# ---------- TAB 2: PASTE MULTI-LINE ----------
with tab_paste:
    st.caption("Dán mỗi dòng là **1 bình luận**. Dòng trống sẽ bị bỏ qua.")
    bulk_text = st.text_area(
        "Dán nhiều bình luận:",
        height=200,
        key="paste_area",
        placeholder="Ví dụ:\nSản phẩm này thật tuyệt vời!\nThằng này nói chuyện mất dạy quá...",
    )
    if st.button("Phân loại danh sách đã dán", key="paste_run"):
        items = [ln.strip() for ln in bulk_text.splitlines() if ln.strip()]
        if not items:
            st.warning("Không có dòng hợp lệ.")
        else:
            with st.spinner("Đang phân tích danh sách..."):
                df_out = _predict_batch(items)

            st.subheader("🔎 Kết quả (có thể sửa nhãn trước khi lưu)")
            edited = st.data_editor(
                df_out,
                hide_index=True,
                column_order=[
                    "raw_text", "processed_text",
                    "pred_label_name", "saved_label_name",
                    "confidence",
                    *[c for c in df_out.columns if c.startswith("proba::")],
                ],
                column_config={
                    "saved_label_name": st.column_config.SelectboxColumn(
                        "saved_label_name",
                        options=["Giữ dự đoán", "Non-toxic", "Toxic"],
                        help="Chọn nhãn sẽ được lưu",
                    ),
                    "confidence": st.column_config.NumberColumn(format="%.2f"),
                },
                num_rows="fixed",
                use_container_width=True,
                key="paste_editor",
            )
            c1, c2 = st.columns([2, 1])
            with c1:
                note_batch = st.text_input("Ghi chú chung (tuỳ chọn):", key="paste_note")
            with c2:
                append_train_batch = st.checkbox("Thêm vào tập huấn luyện", value=True, key="paste_append")
            if st.button("💾 Lưu tất cả hàng", key="paste_save"):
                n_saved = _save_batch(edited, note_batch, append_train_batch)
                st.success(
                    f"Đã lưu {n_saved} hàng vào:\n- Log: `{LOG_PATH}`"
                    + (f"\n- Tập huấn luyện mới: `{TRAIN_APPEND_PATH}`" if append_train_batch else "")
                )

# ---------- TAB 3: MULTI-CSV UPLOAD ----------
with tab_csv:
    st.caption("Tải **1 hoặc nhiều CSV**. Mặc định lấy cột `text`. Có thể đổi tên cột bên dưới.")
    uploads = st.file_uploader(
        "Chọn CSV (có thể nhiều file)", type=["csv"], accept_multiple_files=True, key="csv_uploader"
    )
    col_csv_name = st.text_input("Tên cột chứa bình luận", value="text", key="csv_col_name")
    if st.button("Phân loại CSV đã chọn", key="csv_run"):
        all_rows = []
        file_count = 0
        if not uploads:
            st.warning("Chưa chọn file nào.")
        else:
            for up in uploads:
                try:
                    # ưu tiên utf-8-sig để tránh BOM/UnicodeDecodeError
                    df_in = pd.read_csv(up, encoding="utf-8-sig")
                except Exception:
                    df_in = pd.read_csv(up)
                if col_csv_name not in df_in.columns:
                    st.error(f"File `{getattr(up, 'name', 'CSV')}` không có cột `{col_csv_name}`.")
                    continue
                texts = df_in[col_csv_name].astype(str).tolist()
                texts = [t.strip() for t in texts if str(t).strip()]
                if not texts:
                    continue
                with st.spinner(f"Đang phân tích: {getattr(up, 'name', 'CSV')}"):
                    df_part = _predict_batch(texts)
                    df_part.insert(0, "source_file", getattr(up, "name", "CSV"))
                    all_rows.append(df_part)
                    file_count += 1

            if not all_rows:
                st.warning("Không có dòng hợp lệ để phân loại.")
            else:
                df_all = pd.concat(all_rows, ignore_index=True)
                st.subheader(f"🔎 Kết quả {file_count} file (có thể sửa nhãn trước khi lưu)")
                edited_csv = st.data_editor(
                    df_all,
                    hide_index=True,
                    column_order=[
                        "source_file", "raw_text", "processed_text",
                        "pred_label_name", "saved_label_name", "confidence",
                        *[c for c in df_all.columns if c.startswith("proba::")],
                    ],
                    column_config={
                        "saved_label_name": st.column_config.SelectboxColumn(
                            "saved_label_name",
                            options=["Giữ dự đoán", "Non-toxic", "Toxic"],
                        ),
                        "confidence": st.column_config.NumberColumn(format="%.2f"),
                    },
                    num_rows="fixed",
                    use_container_width=True,
                    key="csv_editor",
                )
                c1, c2 = st.columns([2, 1])
                with c1:
                    note_csv = st.text_input("Ghi chú chung (tuỳ chọn):", key="csv_note")
                with c2:
                    append_train_csv = st.checkbox("Thêm vào tập huấn luyện", value=True, key="csv_append")

                if st.button("💾 Lưu tất cả hàng", key="csv_save"):
                    n_saved = _save_batch(edited_csv, note_csv, append_train_csv)
                    st.success(
                        f"Đã lưu {n_saved} hàng vào:\n- Log: `{LOG_PATH}`"
                        + (f"\n- Tập huấn luyện mới: `{TRAIN_APPEND_PATH}`" if append_train_csv else "")
                    )
