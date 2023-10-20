from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
def Build_Model(model_name):
    if model_name == "xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = Dense(200, activation='sigmoid')(x)  # Assuming you want to predict 200 classes
        model = Model(inputs=base_model.input, outputs=outputs)
        return model
if __name__ == '__main__':
    Tuning_Model()
