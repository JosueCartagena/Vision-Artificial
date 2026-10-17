# 🤟 Reconocimiento del alfabeto de la Lengua de Señas Boliviana (LSB)

Sistema de visión artificial que reconoce en tiempo real, con la cámara web, las letras del alfabeto dactilológico de la **Lengua de Señas Boliviana** y las convierte en texto.

> Proyecto de la carrera de **Ingeniería de Inteligencia Artificial**, UCB Santa Cruz.

---

## ✨ Características

- Reconoce **24 letras estáticas** del alfabeto LSB (A–Y, sin las que requieren movimiento).
- Clase adicional `Sin_letra_o_vacio` para cuando no hay ninguna seña frente a la cámara.
- Modelo entrenado con **Teachable Machine** y convertido a **ONNX** para una inferencia ligera.
- Interfaz de escritorio (**PySide6**) que va armando el texto: una letra se escribe cuando se mantiene estable unos ~0,4 s.
- Se puede empaquetar como `.exe` para Windows.
- Modo de prueba sin cámara: `python lsb_app.py --demo foto.jpg`.

## 🗂️ Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| [`01_modelo/`](01_modelo) | Modelo entrenado: `keras_model.h5`, `model.onnx` (el que usa la app), `labels.txt` y `tm_loader.py` (carga del `.h5` con TensorFlow nuevo). |
| [`02_dataset/`](02_dataset) | `proyecto_nuevo_v9.tm` (7.410 imágenes, 300 por clase) y `codigo_generador/` con los scripts que crearon el dataset. |
| [`03_app_ejecutable/`](03_app_ejecutable) | Las dos aplicaciones (ver abajo). |
| [`04_paper/`](04_paper) | Artículo científico en Word y PDF. |
| [`05_informes/`](05_informes) | Informes del proyecto y guías de apoyo (qué señas usar, palabras en quechua, pasos de la demo). |
| [`06_presentacion/`](06_presentacion) | Presentación web: abre `index.html` en el navegador y avanza con las flechas. |

## 🚀 Cómo usarlo

Requisito: **Python 3.10 – 3.12** y una cámara web.

### Opción A: app principal (PySide6 + ONNX) — recomendada

```bash
cd 03_app_ejecutable/app_PySide6_ONNX
pip install -r requirements.txt
python lsb_app.py
```

En Windows también puedes hacer doble clic en `ejecutar_sin_exe.bat`.

**Crear el `.exe` (solo Windows):** doble clic en `crear_exe.bat`. El programa queda en `dist\LSB_Reconocimiento\LSB_Reconocimiento.exe`. Para llevarlo a otra PC copia **toda** la carpeta `dist\LSB_Reconocimiento`, no solo el `.exe`.

### Opción B: versión simple (Tkinter + Keras)

```bash
cd 03_app_ejecutable/app_Tkinter_Keras
pip install -r requirements.txt
python prueba_camara.py     # prueba rápida de cámara (Q para salir)
python interfaz.py          # interfaz gráfica
```

## 🧠 Sobre el modelo

- **Entrenamiento:** Teachable Machine (clasificación de imágenes).
- **Clases:** 25 (24 letras + `Sin_letra_o_vacio`).
- **Dataset:** 7.410 imágenes, 300 por clase. Para reentrenar, abre `02_dataset/proyecto_nuevo_v9.tm` en Teachable Machine con *Abrir proyecto desde archivo*.
- **Formatos:** `keras_model.h5` (original) y `model.onnx` (convertido, mismas predicciones).

Letras reconocidas: `A B C D E F G H I K L M N O P Q R S T U V W X Y`

## 📊 Resultados y limitaciones

- Precisión medida: **87 %** en imágenes originales y **100 %** en imágenes sintéticas. Esta última usa datos de entrenamiento, así que es optimista.
- **No hay aún evaluación con cámara real.** La interfaz solo se probó sin cámara.
- Las letras con movimiento (**CH, J, LL, Ñ, RR, Z**) no están soportadas: quedan como trabajo futuro.

## 🔭 Trabajo futuro

- [ ] Evaluar el sistema con cámara real y con distintas personas, fondos e iluminación.
- [ ] Soportar las letras con movimiento (CH, J, LL, Ñ, RR, Z).
- [ ] Reconocer palabras y frases completas.
- [ ] Validar el vocabulario en quechua y aimara con hablantes nativos.
- [ ] Publicar el `.exe` compilado como *Release*.

## 📄 Documentación

- Artículo: [`04_paper/Articulo_LSB_final.pdf`](04_paper/Articulo_LSB_final.pdf)
- Informe resumido: [`05_informes/Informe_proyecto_LSB_resumen.pdf`](05_informes/Informe_proyecto_LSB_resumen.pdf)
- Presentación publicada: https://claude.ai/artifact/25ySZrtLpmcXCXyj4nne9A

## 👤 Autor

**Josué Cartagena** — UCB Santa Cruz, Ingeniería de Inteligencia Artificial.
