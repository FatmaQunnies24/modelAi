from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import numpy as np
import tensorflow as tf
from PIL import Image

app = FastAPI()

# افترض أن لديك نموذجين:
# 1) النموذج الأول لتصنيف نوع البشرة (dry, normal, oily)
skin_type_model = tf.keras.models.load_model('data/Models/DryNormalOily.h5')

# 2) النموذج الثاني لتقدير نسب (التصبغات، التجاعيد، حب الشباب، والبشرة الطبيعية)
details_model = tf.keras.models.load_model('data/Models/ShrinkSpotAcneClear.h5')

# مسار لحفظ الصور في حال رغبت بتخزينها
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def preprocess_image(image_file: UploadFile) -> np.ndarray:
    """دالة لتحويل الصورة إلى مصفوفة قابلة للإدخال للنموذج."""
    image = Image.open(image_file.file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # إضافة بُعد الدفعة (batch)
    return image_array


@app.post("/analyze/")
async def analyze_skin_type(skin_type: UploadFile = File(...)):
    """
    المسار الأول لتحليل نوع البشرة (dry, normal, oily).
    """
    if not skin_type.filename:
        return {"error": "لم يتم رفع صورة للبشرة"}

    # معالجة الصورة
    skin_type_image = preprocess_image(skin_type)
    # توقع النوع باستخدام النموذج الأول
    prediction = skin_type_model.predict(skin_type_image)
    predicted_class = np.argmax(prediction)

    # نفترض أن ترتيب الفئات في النموذج: 0 = dry, 1 = normal, 2 = oily
    labels = {0: "dry", 1: "normal", 2: "oily"}
    result_label = labels.get(predicted_class, "unknown")

    return {
        "skin_type": result_label
    }


@app.post("/analyze_details/")
async def analyze_skin_details(details_file: UploadFile = File(...)):
    """
    المسار الثاني لتحليل تفاصيل البشرة (نسب التصبغات، التجاعيد، حب الشباب، والبشرة الطبيعية).
    """
    if not details_file.filename:
        return {"error": "لم يتم رفع صورة للبشرة"}

    # معالجة الصورة
    details_image = preprocess_image(details_file)
    # افترض أن النموذج الثاني يُرجع 4 احتمالات بالترتيب:
    # [pigmentation, wrinkles, acne, normal]
    details_prediction = details_model.predict(details_image)[0]  # مصفوفة بطول 4

    # تحويل القيم إلى نسب مئوية
    # مثلاً إذا كان details_prediction = [0.2, 0.3, 0.1, 0.4]
    # ستكون النسب: 20%, 30%, 10%, 40%
    percentages = details_prediction * 100

    return {
        "pigmentation": round(float(percentages[0]), 2),
        "wrinkles": round(float(percentages[1]), 2),
        "acne": round(float(percentages[2]), 2),
        "normal": round(float(percentages[3]), 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
