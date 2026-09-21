# app.py — เว็บแอปทำนายการรอดชีวิตจาก Titanic ด้วยโมเดล scikit-learn
# วิธีรัน:  streamlit run app.py

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ (ต้องเป็นคำสั่ง st.* คำสั่งแรกของไฟล์)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# ชื่อไฟล์โมเดล — ต้องวางไว้โฟลเดอร์เดียวกับ app.py
MODEL_PATH = Path(__file__).parent / "titanic_model.joblib"

# ฟีเจอร์ที่โมเดลต้องการ เรียงลำดับให้ตรงกับตอนเทรน (ห้ามสลับ)
FEATURE_ORDER = ["Pclass", "Sex_female", "Age", "Fare", "FamilySize"]


# ---------------------------------------------------------------------------
# โหลดโมเดลครั้งเดียวแล้วเก็บไว้ในแคช (เร็วขึ้น ไม่ต้องโหลดใหม่ทุกครั้งที่กดปุ่ม)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# ถ้าหาไฟล์โมเดลไม่เจอ ให้แจ้งเตือนแล้วหยุด
if model is None:
    st.error(
        f"❌ ไม่พบไฟล์โมเดล: {MODEL_PATH.name}\n\n"
        "กรุณาวางไฟล์ `titanic_model.joblib` ไว้ในโฟลเดอร์เดียวกับ app.py"
    )
    st.stop()


# ---------------------------------------------------------------------------
# ส่วนหัวของหน้า
# ---------------------------------------------------------------------------
st.title("🚢 Titanic Survival Predictor")
st.write(
    "กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกดปุ่มเพื่อให้โมเดลทำนายว่า "
    "**รอดชีวิต** หรือไม่"
)

# ---------------------------------------------------------------------------
# ฟอร์มรับข้อมูล — รวมอินพุตไว้ในฟอร์ม เพื่อให้โมเดลทำนายตอนกดปุ่มเท่านั้น
# ---------------------------------------------------------------------------
with st.form("passenger_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox(
            "ชั้นโดยสาร (Pclass)",
            options=[1, 2, 3],
            index=2,  # ค่าเริ่มต้น = ชั้น 3
            help="1 = ชั้นหนึ่ง, 2 = ชั้นสอง, 3 = ชั้นสาม",
        )
        sex = st.radio("เพศ (Sex)", options=["หญิง", "ชาย"], horizontal=True)
        age = st.slider("อายุ (Age)", min_value=0, max_value=100, value=30)

    with col2:
        fare = st.number_input(
            "ค่าโดยสาร (Fare)",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0,
            help="ค่าตั๋วโดยสาร (หน่วยปอนด์)",
        )
        family_size = st.number_input(
            "ขนาดครอบครัวบนเรือ (FamilySize)",
            min_value=1,
            max_value=15,
            value=1,
            step=1,
            help="นับตัวเองด้วย: 1 = เดินทางคนเดียว",
        )

    submitted = st.form_submit_button("🔮 ทำนายผล")

# ---------------------------------------------------------------------------
# เมื่อกดปุ่ม: เตรียมข้อมูล → ทำนาย → แสดงผล
# ---------------------------------------------------------------------------
if submitted:
    # แปลงเพศเป็นรหัสตัวเลขให้ตรงกับที่โมเดลต้องการ (หญิง = 1, ชาย = 0)
    sex_female = 1 if sex == "หญิง" else 0

    # สร้าง DataFrame หนึ่งแถว โดยใช้ชื่อคอลัมน์และลำดับให้ตรงกับตอนเทรน
    input_df = pd.DataFrame(
        [[pclass, sex_female, age, fare, family_size]],
        columns=FEATURE_ORDER,
    )

    # ทำนายผล
    prediction = model.predict(input_df)[0]          # 0 = ไม่รอด, 1 = รอด
    proba = model.predict_proba(input_df)[0]         # [ความน่าจะเป็นไม่รอด, รอด]
    survive_chance = proba[1]                        # ความน่าจะเป็นที่จะรอด

    st.divider()

    if prediction == 1:
        st.success("### ✅ ทำนายว่า: รอดชีวิต")
    else:
        st.error("### ❌ ทำนายว่า: ไม่รอดชีวิต")

    # แสดงความน่าจะเป็นเป็นเปอร์เซ็นต์ + แถบความคืบหน้า
    st.metric("โอกาสรอดชีวิต", f"{survive_chance * 100:.1f}%")
    st.progress(float(survive_chance))

    # แสดงข้อมูลที่ป้อนเข้าไป (ไว้ตรวจทาน)
    with st.expander("ดูข้อมูลที่ใช้ทำนาย"):
        st.dataframe(input_df, use_container_width=True)
