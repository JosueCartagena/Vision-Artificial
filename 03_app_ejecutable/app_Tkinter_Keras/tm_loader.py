"""Carga el modelo exportado de Teachable Machine (Keras .h5) con TensorFlow 2.16+ / Keras 3."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS","1")   # Keras 2 (paquete tf-keras): el modelo de TM no carga en Keras 3
import numpy as np, tensorflow as tf

class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *a, **k):
        k.pop("groups", None)          # TM lo exporta con 'groups', Keras nuevo no lo acepta
        super().__init__(*a, **k)

def load(model_path="keras_model.h5", labels_path="labels.txt"):
    model = tf.keras.models.load_model(model_path, compile=False,
                                       custom_objects={"DepthwiseConv2D": DepthwiseConv2D})
    labels = [l.strip().split(" ", 1)[1] if " " in l else l.strip() for l in open(labels_path, encoding="utf-8") if l.strip()]
    return model, labels

def preprocess(bgr):
    """Igual que Teachable Machine: recorte cuadrado al centro, 224x224, RGB, rango [-1, 1]."""
    import cv2
    h, w = bgr.shape[:2]; s = min(h, w); y, x = (h - s) // 2, (w - s) // 2
    im = cv2.resize(bgr[y:y+s, x:x+s], (224, 224), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (im / 127.5 - 1.0)[None]

def predict(model, labels, bgr):
    p = model.predict(preprocess(bgr), verbose=0)[0]
    i = int(p.argmax()); return labels[i], float(p[i]), p
