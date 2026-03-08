import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================
# 1. จัดการ Path และโหลดข้อมูล
# ==========================================
SURVEY_CSV = 'data/data_from_questionaire.csv'
IG_CSV = 'data/data_from_intragram.csv'

SURVEY_IMG_DIR = 'data/questionnaire_images/'
IG_IMG_DIR = 'data/instagram_images/'

# ตรวจสอบว่ามีไฟล์ CSV อยู่จริงหรือไม่
if not os.path.exists(SURVEY_CSV) or not os.path.exists(IG_CSV):
    print("❌ ไม่พบไฟล์ CSV กรุณาตรวจสอบโฟลเดอร์ data/")
    exit()

print("กำลังโหลดและเตรียมข้อมูล...")
df_survey = pd.read_csv(SURVEY_CSV)
df_ig = pd.read_csv(IG_CSV)

# สร้าง Full Path ให้รูปภาพแบบสอบถาม (อยู่ในโฟลเดอร์รวม)
df_survey['Image 1'] = SURVEY_IMG_DIR + df_survey['Image 1']
df_survey['Image 2'] = SURVEY_IMG_DIR + df_survey['Image 2']

# สร้าง Full Path ให้รูปภาพ Instagram (ต้องแทรกชื่อ Menu เข้าไปใน Path ด้วย)
df_ig['Image 1'] = IG_IMG_DIR + df_ig['Menu'] + '/' + df_ig['Image 1']
df_ig['Image 2'] = IG_IMG_DIR + df_ig['Menu'] + '/' + df_ig['Image 2']

# รวม Data เข้าด้วยกัน
df_all = pd.concat([df_survey[['Image 1', 'Image 2', 'Menu', 'Winner']], 
                    df_ig[['Image 1', 'Image 2', 'Menu', 'Winner']]], ignore_index=True)

# แปลง Label: Winner 1 -> 0 (โมเดลทาย < 0.5), Winner 2 -> 1 (โมเดลทาย >= 0.5)
df_all['label'] = df_all['Winner'].apply(lambda x: 0 if x == 1 else 1)

# แบ่ง Train/Val (ป้องกัน Leakage และรักษาสัดส่วน Menu)
train_df, val_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['Menu'])
print(f"✅ ข้อมูลพร้อม! ชุด Train: {len(train_df)} คู่ | ชุด Validation: {len(val_df)} คู่")

# ==========================================
# 2. สร้าง tf.data.Dataset Pipeline
# ==========================================
def load_and_preprocess_image(path1, path2, label):
    def process_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img
    
    img1 = process_img(path1)
    img2 = process_img(path2)
    # ชื่อ Key ต้องตรงกับชื่อ Input Layer ของโมเดลเป๊ะๆ
    return ({"image_1": img1, "image_2": img2}, label)

def create_dataset(dataframe, batch_size=32, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe['Image 1'].values,
        dataframe['Image 2'].values,
        dataframe['label'].values
    ))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

BATCH_SIZE = 32
train_dataset = create_dataset(train_df, batch_size=BATCH_SIZE, is_training=True)
val_dataset = create_dataset(val_df, batch_size=BATCH_SIZE, is_training=False)

# ==========================================
# 3. สร้างสถาปัตยกรรม Siamese Network
# ==========================================
print("กำลังสร้างสถาปัตยกรรมโมเดล...")
def build_siamese_model(input_shape=(224, 224, 3)):
    # โหลด Pre-trained model (EfficientNetB0) มาเป็นตัวสกัดจุดเด่นภาพ
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze น้ำหนักไว้ก่อนในช่วงแรก เพื่อไม่ให้ความรู้เดิมพัง
    base_model.trainable = False 
    
    inputs = Input(shape=input_shape)
    # ปิดโหมด training เพื่อล็อกค่า BatchNormalization
    x = base_model(inputs, training=False) 
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs, x, name="feature_extractor")

    # กำหนด Input 2 ทาง (ภาพ 1 และ ภาพ 2)
    input_1 = Input(shape=input_shape, name="image_1")
    input_2 = Input(shape=input_shape, name="image_2")
    
    # นำภาพผ่านตัวสกัดจุดเด่นตัวเดียวกัน
    feat_1 = feature_extractor(input_1)
    feat_2 = feature_extractor(input_2)
    
    # นำจุดเด่นมาต่อกันแล้วตัดสินใจ
    merged = Concatenate()([feat_1, feat_2])
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    
    # ออกเป็นค่าความน่าจะเป็น 0 ถึง 1
    outputs = Dense(1, activation='sigmoid', name="output")(x)
    
    return Model(inputs=[input_1, input_2], outputs=outputs, name="siamese_network")

model = build_siamese_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ==========================================
# 4. สั่งเทรนโมเดล (Training Loop)
# ==========================================
# 📌 เปลี่ยนมาใช้นามสกุล .keras เพื่อป้องกันบั๊กเวลาโหลดโมเดลกลับมาใช้ใน Keras 3
MODEL_SAVE_PATH = 'best_siamese_model.keras'

callbacks = [
    # เซฟเฉพาะเวอร์ชันที่ Validation Accuracy สูงที่สุด
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    # หยุดเทรนอัตโนมัติถ้า 5 Epochs ผ่านไปแล้วโมเดลไม่เก่งขึ้น
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
]

print("🚀 เริ่มฝึกสอนโมเดล (Training)...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20, # ปรับจำนวนรอบเพิ่ม/ลดได้ตามต้องการ
    callbacks=callbacks
)

print(f"🎉 เทรนเสร็จสิ้น! โมเดลที่แม่นที่สุดถูกเซฟไว้ในชื่อ '{MODEL_SAVE_PATH}'")