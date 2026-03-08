import pandas as pd
import tensorflow as tf
import os

# ==========================================
# 1. โหลดข้อมูลที่มีอยู่มาทดสอบ (สุ่มมา 10 คู่)
# ==========================================
CSV_PATH = 'data/data_from_questionaire.csv'
IMG_DIR = 'data/questionnaire_images/'

df_test = pd.read_csv(CSV_PATH)
df_sample = df_test.sample(500, random_state=42).copy()

img1_paths = IMG_DIR + df_sample['Image 1']
img2_paths = IMG_DIR + df_sample['Image 2']

# ==========================================
# 2. สร้าง Data Pipeline
# ==========================================
def process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return img

def preprocess_test_data(path1, path2):
    img1 = process_img(path1)
    img2 = process_img(path2)
    return {"image_1": img1, "image_2": img2}

test_dataset = tf.data.Dataset.from_tensor_slices((img1_paths.values, img2_paths.values))
test_dataset = test_dataset.map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE).batch(10)

# ==========================================
# 3. โหลดโมเดลแบบคลีนๆ (ไม่ต้องสร้างโครงสร้างรอแล้ว)
# ==========================================
MODEL_PATH = 'best_siamese_model.keras' 

if not os.path.exists(MODEL_PATH):
    print(f"❌ ไม่พบไฟล์โมเดล {MODEL_PATH}")
    exit()

print(f"กำลังโหลดโมเดล {MODEL_PATH} ...")
# โหลดรวดเดียวจบ!
model = tf.keras.models.load_model(MODEL_PATH)

print("🤖 กำลังวิเคราะห์รูปภาพ...")
predictions = model.predict(test_dataset)

# ==========================================
# 4. สรุปผลลัพธ์
# ==========================================
predicted_winners = [1 if prob[0] < 0.5 else 2 for prob in predictions]
probabilities = [prob[0] for prob in predictions]

df_sample['Predicted'] = predicted_winners
df_sample['Prob'] = probabilities
df_sample['Correct?'] = df_sample['Winner'] == df_sample['Predicted']

print("\n" + "="*50)
print("🎯 ผลการทดสอบ (สุ่มมา 10 คู่)")
print("="*50)
print(df_sample[['Image 1', 'Image 2', 'Winner', 'Predicted', 'Correct?']])

accuracy = df_sample['Correct?'].mean() * 100
print(f"\n✅ ความแม่นยำจากกลุ่มตัวอย่างนี้: {accuracy:.2f}%")