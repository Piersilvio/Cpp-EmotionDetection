# ----------------------------
# author: Onorati Jacopo
#
# This script was written to convert the CNN
# model outputed in .keras format by running the notebook at
# https://www.kaggle.com/code/abhirupkumarbhowmick/emotion-detection
# The file obtained is in .onnx format.
#
# Requirements:
# Python: 3.11.13
# TensorFlow: 2.20.0
# Keras: 3.11.3
# ONNX: 1.17.0
# ----------------------------

import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
import onnx

# ----------------------------
# def. of the custom layers
# ----------------------------
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(self.channels // self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.channels, activation='sigmoid')

    def call(self, inputs):
        se = self.squeeze(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        # We use the statically defined channel size
        se = tf.reshape(se, [-1, 1, 1, self.channels])
        return inputs * se

    # for the serialization
    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
        })
        return config

class CAMBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(CAMBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.dense1 = tf.keras.layers.Dense(self.channels // self.reduction_ratio, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.channels, activation='sigmoid')

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)
        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))
        out = avg_out + max_out
        
        out = tf.reshape(out, [-1, 1, 1, self.channels])
        return inputs * out

    def get_config(self):
        config = super(CAMBlock, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
        })
        return config

if __name__ == "__main__":
    # ----------------------------
    # load the .keras model
    # ----------------------------
    model = load_model("Fine_tuned_CNN_model.keras", custom_objects={
        "SEBlock": SEBlock,
        "CAMBlock": CAMBlock
    })

    model.summary()

    # ----------------------------
    # conversion in ONNX
    # ----------------------------
    input_signature = [
        tf.TensorSpec(shape=[None, 224, 224, 1], dtype=tf.float32, name="input")
    ]

    # setting the opset=1313
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    # ----------------------------
    # saving the model in .onnx
    # ----------------------------
    onnx.save_model(onnx_model, "Fine_tuned_CNN_model.onnx")
    print("Modello convertito e salvato in ONNX con successo!")