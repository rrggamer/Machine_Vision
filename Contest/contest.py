import pandas as pd
import tensorflow as tf
import os

# ==========================================
# 1. ตั้งค่า Path ไฟล์ตามที่โจทย์กำหนด
# ==========================================
# ตอนสอบจริง โฟลเดอร์จะชื่อ "Test Images" และไฟล์ชื่อ "test.csv" 
TEST_CSV_PATH = 'test.csv'  
TEST_IMG_DIR = 'Test Images/' 

# ตรวจสอบว่ามีไฟล์ทดสอบอยู่จริงหรือไม่
if not os.path.exists(TEST_CSV_PATH):
    print(f"❌ ไม่พบไฟล์ {TEST_CSV_PATH} กรุณาตรวจสอบให้แน่ใจว่าวางไฟล์ไว้ถูกที่")
    exit()

df_test = pd.read_csv(TEST_CSV_PATH)
print(f"โหลดข้อมูลทดสอบมาทั้งหมด: {len(df_test)} คู่")

# ==========================================
# 2. เตรียมข้อมูล (Data Pipeline สำหรับ Predict)
# ==========================================
def process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return img

def preprocess_test_data(path1, path2):
    img1 = process_img(path1)
    img2 = process_img(path2)
    # คืนค่าเป็น Dictionary ให้ตรงกับชื่อ Input Layer ของโมเดล Siamese
    return {"image_1": img1, "image_2": img2}

# สร้าง Full Path สำหรับอ่านไฟล์รูปภาพตอนสอบ
# ตรวจสอบให้แน่ใจว่า path รูปภาพประกอบด้วยโฟลเดอร์ Test Images และตามด้วยชื่อไฟล์
img1_paths = TEST_IMG_DIR + df_test['Image 1']
img2_paths = TEST_IMG_DIR + df_test['Image 2']

# สร้าง Dataset สำหรับป้อนเข้าโมเดล (ไม่มี Label แล้วเพราะเราต้องทำนายเอง)
test_dataset = tf.data.Dataset.from_tensor_slices((img1_paths.values, img2_paths.values))
test_dataset = test_dataset.map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 3. โหลดโมเดลและทำนายผล
# ==========================================
# เปลี่ยนชื่อไฟล์ให้ตรงกับโมเดลที่คุณรันได้ดีที่สุด
MODEL_PATH = 'best_siamese_model.keras' 

if not os.path.exists(MODEL_PATH):
    print(f"❌ ไม่พบไฟล์โมเดล {MODEL_PATH} กรุณาตรวจสอบชื่อไฟล์")
    exit()

print(f"กำลังโหลดโมเดล {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)

print("🤖 กำลังวิเคราะห์รูปภาพ...")
predictions = model.predict(test_dataset)

# ==========================================
# 4. แปลงผลลัพธ์และบันทึกไฟล์ส่ง
# ==========================================
# โมเดลเราใช้ Sigmoid พ่นค่าความน่าจะเป็น 0 ถึง 1
# < 0.5 คือโมเดลทายว่า Label 0 (Image 1 ชนะ) -> เปลี่ยนคำตอบเป็น 1
# >= 0.5 คือโมเดลทายว่า Label 1 (Image 2 ชนะ) -> เปลี่ยนคำตอบเป็น 2
final_answers = [1 if prob[0] < 0.5 else 2 for prob in predictions]

# อัปเดตคำตอบลงในคอลัมน์ Winner (แทนที่เลข 0 เดิม) 
df_test['Winner'] = final_answers

# เซฟทับไฟล์เดิมเพื่อเตรียมส่ง
df_test.to_csv(TEST_CSV_PATH, index=False)
print(f"✅ ทำนายผลเสร็จสิ้น! บันทึกลง {TEST_CSV_PATH} เรียบร้อย ตรวจสอบไฟล์และส่งใน Google Form ได้เลยครับ")