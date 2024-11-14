from keras.saving.save import load_model
import MakeDataset as md
import ModelS as ms
import ModelPredict as mp
import os
import Config
import datetime
import ModelTrain as mt
import hfloss as hf

input1_3D, input2_3D, output_3D = md.get_datasets()
tr_dataset, val_dataset = md.create_tf_dataset(input1_3D, input2_3D, output_3D, batch_size=6)
print("TensorFlow dataset created.")

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# model_save_path = f'./unwrapnet_{current_time}.h5'
model_save_path = f'./unwrapnet_{current_time}.h5'

if not os.path.exists(model_save_path):
    model = ms.unwrapnet()
    model.summary()
    history = mt.train_model(model,tr_dataset,val_dataset, epochs=Config.epoch, batch_size=Config.batch_size, checkpoint_path='best_uwwrap_model.h5')
    model.save(model_save_path)
    print(f"UNet model saved to {model_save_path}")

if os.path.exists(model_save_path):
    model = load_model(model_save_path, custom_objects={'high_fidelity_loss': hf.high_fidelity_loss})

for i in range(3):
    mp.predict_and_display(model, input1_3D, input2_3D, output_3D)