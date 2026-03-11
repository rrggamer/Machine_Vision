import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB3 # อัปเกรด Backbone
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D, 
                                     Concatenate, Dropout, Lambda, Multiply,
                                     RandomFlip, RandomRotation, RandomZoom,
                                     RandomBrightness, RandomContrast, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# ==========================================
# 1. จัดการ Path และโหลดข้อมูล (ใช้แบบเดิม)
# ==========================================
SURVEY_CSV = 'data/data_from_questionaire.csv'
IG_CSV = 'data/data_from_intragram.csv'

SURVEY_IMG_DIR = 'data/questionnaire_images/'
IG_IMG_DIR = 'data/instagram_images/'

if not os.path.exists(SURVEY_CSV) or not os.path.exists(IG_CSV):
    print("❌ ไม่พบไฟล์ CSV กรุณาตรวจสอบโฟลเดอร์ data/")
    exit()

df_survey = pd.read_csv(SURVEY_CSV)
df_ig = pd.read_csv(IG_CSV)

df_survey['Image 1'] = SURVEY_IMG_DIR + df_survey['Image 1']
df_survey['Image 2'] = SURVEY_IMG_DIR + df_survey['Image 2']
df_ig['Image 1'] = IG_IMG_DIR + df_ig['Menu'] + '/' + df_ig['Image 1']
df_ig['Image 2'] = IG_IMG_DIR + df_ig['Menu'] + '/' + df_ig['Image 2']

df_all = pd.concat([df_survey[['Image 1', 'Image 2', 'Menu', 'Winner']], 
                    df_ig[['Image 1', 'Image 2', 'Menu', 'Winner']]], ignore_index=True)

df_all['label'] = df_all['Winner'].apply(lambda x: 0 if x == 1 else 1)
train_df, val_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['Menu'])

# ==========================================
# 2. Data Swapping (เบิ้ลข้อมูลเป็น 2 เท่า)
# ==========================================
def augment_by_swapping(df):
    df_swapped = df.copy()
    temp = df_swapped['Image 1'].copy()
    df_swapped['Image 1'] = df_swapped['Image 2']
    df_swapped['Image 2'] = temp
    df_swapped['label'] = 1 - df_swapped['label']
    return pd.concat([df, df_swapped], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

train_df = augment_by_swapping(train_df)
val_df = augment_by_swapping(val_df)
print(f"🔥 ข้อมูลพร้อมปะทะ! Train: {len(train_df)} คู่ | Validation: {len(val_df)} คู่")

# ==========================================
# 3. สร้าง tf.data.Dataset Pipeline
# ==========================================
def load_and_preprocess_image(path1, path2, label):
    def process_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return img
    return ({"image_1": process_img(path1), "image_2": process_img(path2)}, label)

def create_dataset(dataframe, batch_size=16, is_training=False): # ลด Batch Size ป้องกันการ์ดจอระเบิด
    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe['Image 1'].values, dataframe['Image 2'].values, dataframe['label'].values
    ))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ⚠️ ปรับ Batch Size เป็น 16 เพราะ B3 กินแรมเยอะกว่า B0
BATCH_SIZE = 16 
train_dataset = create_dataset(train_df, batch_size=BATCH_SIZE, is_training=True)
val_dataset = create_dataset(val_df, batch_size=BATCH_SIZE, is_training=False)

# ==========================================
# 4. สร้างสถาปัตยกรรมระดับ Heavyweight
# ==========================================
print("กำลังประกอบร่างสถาปัตยกรรม Hardcore Siamese (EfficientNetB3)...")

# Augmentation แบบจัดเต็ม (เพิ่มแสงและคอนทราสต์)
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomBrightness(factor=0.2),
    RandomContrast(factor=0.2),
], name="heavy_augmentation")

def build_hardcore_siamese(input_shape=(224, 224, 3)):
    # โหลด EfficientNetB3
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Fine-tune ลึกขึ้น (เปิดให้เทรน 30 เลเยอร์สุดท้าย)
    base_model.trainable = True
    for layer in base_model.layers[:-30]: 
        layer.trainable = False 
    
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False) 
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs, x, name="feature_extractor")

    input_1 = Input(shape=input_shape, name="image_1")
    input_2 = Input(shape=input_shape, name="image_2")
    
    feat_1 = feature_extractor(input_1)
    feat_2 = feature_extractor(input_2)
    
    # 🔥 Deep Feature Fusion: ผสมผสานคณิตศาสตร์ขั้นสูง
    # 1. ระยะห่างสัมบูรณ์ (Absolute Difference)
    l1_dist = Lambda(lambda t: K.abs(t[0] - t[1]), name="l1_distance")([feat_1, feat_2])
    # 2. ผลคูณ (Element-wise Multiplication)
    multiply = Multiply(name="multiply")([feat_1, feat_2])
    
    # รวมทุกท่าเข้าด้วยกัน (ให้โมเดลมีข้อมูลไปตัดสินใจเยอะที่สุด)
    merged = Concatenate()([feat_1, feat_2, l1_dist, multiply])
    
    # 🧠 Super Head: ใช้ BatchNormalization ช่วยให้เทรนได้เร็วและลึกขึ้น
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

model = build_hardcore_siamese()

# ใช้ Learning Rate ต่ำๆ เพราะโมเดลใหญ่มาก (ป้องกัน Loss แกว่ง)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# 5. สั่งเทรนโมเดลแบบมาราธอน
# ==========================================
MODEL_SAVE_PATH = 'best_siamese_model_hardcore.keras'

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
    # เพิ่มความอดทน (Patience) เป็น 15 Epochs ปล่อยให้มันค่อยๆ เรียนรู้
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

print("🚀 เริ่มฝึกสอนโมเดล (Training)... เตรียมตัวรอกันยาวๆ!")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50, # ปรับลากยาวไป 50 รอบเลย
    callbacks=callbacks
)

print(f"🎉 เทรนเสร็จสิ้น! บอสใหญ่ถูกเซฟไว้ในชื่อ '{MODEL_SAVE_PATH}'")