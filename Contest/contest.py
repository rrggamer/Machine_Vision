import pandas as pd
import tensorflow as tf
import os
import tensorflow.keras.backend as K
from tensorflow.keras.applications import EfficientNetB3 # เปลี่ยนมาโหลด B3
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D, 
                                     Concatenate, Dropout, Lambda, Multiply,
                                     RandomFlip, RandomRotation, RandomZoom,
                                     RandomBrightness, RandomContrast, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# ==========================================
# 1. ตั้งค่า Path ไฟล์
# ==========================================
TEST_CSV_PATH = 'test.csv'  
TEST_IMG_DIR = 'Test Images/' 

if not os.path.exists(TEST_CSV_PATH):
    print(f"❌ ไม่พบไฟล์ {TEST_CSV_PATH}")
    exit()

df_test = pd.read_csv(TEST_CSV_PATH)
print(f"โหลดข้อมูลทดสอบมาทั้งหมด: {len(df_test)} คู่")

# ==========================================
# 2. เตรียมข้อมูล 
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

img1_paths = TEST_IMG_DIR + df_test['Image 1']
img2_paths = TEST_IMG_DIR + df_test['Image 2']

test_dataset = tf.data.Dataset.from_tensor_slices((img1_paths.values, img2_paths.values))
test_dataset = test_dataset.map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(16).prefetch(tf.data.AUTOTUNE) # ปรับ Batch ให้เท่าตอน Train

# ==========================================
# 3. สร้างโครงสร้างโมเดล (ร่าง Hardcore)
# ==========================================
print("กำลังสร้างสถาปัตยกรรม Hardcore Siamese รอรับ Weights...")

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomBrightness(factor=0.2),
    RandomContrast(factor=0.2),
], name="heavy_augmentation")

def build_hardcore_siamese(input_shape=(224, 224, 3)):
    base_model = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape) # ไม่ต้องโหลด imagenet แล้วเพราะเดี๋ยวเราโหลดทับ
    
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False) 
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs, x, name="feature_extractor")

    input_1 = Input(shape=input_shape, name="image_1")
    input_2 = Input(shape=input_shape, name="image_2")
    
    feat_1 = feature_extractor(input_1)
    feat_2 = feature_extractor(input_2)
    
    # Fusion ชุดเดียวกับตอน Train
    l1_dist = Lambda(lambda t: K.abs(t[0] - t[1]), name="l1_distance")([feat_1, feat_2])
    multiply = Multiply(name="multiply")([feat_1, feat_2])
    merged = Concatenate()([feat_1, feat_2, l1_dist, multiply])
    
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.005))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid', name="output")(x)
    
    return Model(inputs=[input_1, input_2], outputs=outputs, name="hardcore_siamese")

# สร้างโมเดลเปล่าๆ
model = build_hardcore_siamese()

# ==========================================
# 4. โหลด Weights และทำนายผล
# ==========================================
MODEL_PATH = 'best_siamese_model_hardcore.keras' # ชี้เป้าไปที่บอสใหญ่

if not os.path.exists(MODEL_PATH):
    print(f"❌ ไม่พบไฟล์โมเดล {MODEL_PATH}")
    exit()

print(f"กำลังโหลดน้ำหนักจาก {MODEL_PATH} ...")
model.load_weights(MODEL_PATH)

print("🤖 กำลังวิเคราะห์รูปภาพ...")
predictions = model.predict(test_dataset)

# ==========================================
# 5. แปลงผลลัพธ์และบันทึกไฟล์ส่ง
# ==========================================
final_answers = [1 if prob[0] < 0.5 else 2 for prob in predictions]
df_test['Winner'] = final_answers
df_test.to_csv(TEST_CSV_PATH, index=False)

print(f"✅ ทำนายผลเสร็จสิ้น! โหดสมชื่อ บันทึกลง {TEST_CSV_PATH} เรียบร้อยครับ")