PROYECTO LSB - Reconocimiento del alfabeto de la Lengua de Señas Boliviana
UCB Santa Cruz - Ingeniería de Inteligencia Artificial

CARPETAS
01_modelo/            Modelo entrenado en Teachable Machine: keras_model.h5, labels.txt,
                      model.onnx (convertido, usado por la app) y tm_loader.py (carga el .h5).
02_dataset/           proyecto_nuevo_v9.tm (7.410 imágenes, 300 por clase, abrir en Teachable Machine
                      con "Abrir proyecto desde archivo") y codigo_generador/ (scripts que crearon el dataset).
03_app_ejecutable/    app_PySide6_ONNX/  -> app principal. ejecutar_sin_exe.bat para probarla;
                                           crear_exe.bat genera LSB_Reconocimiento.exe (SOLO en Windows).
                      app_Tkinter_Keras/ -> versión simple de prueba (prueba_camara.py, interfaz.py).
04_paper/             Artículo en Word y PDF.
05_informes/          Informes para IA (completo y resumen), texto corto, y PDFs de apoyo
                      (qué señas usar, palabras en quechua, pasos de la demo).
06_presentacion/      index.html + img/ (abrir index.html en el navegador; versión publicada en claude.ai:
                      https://claude.ai/artifact/25ySZrtLpmcXCXyj4nne9A). Flechas / deslizar para avanzar.

PENDIENTE / A TENER EN CUENTA
- El .exe NO está compilado: hay que ejecutar crear_exe.bat en Windows (necesita Python y requirements.txt).
- La interfaz solo se probó sin cámara (captura offscreen); falta probarla con cámara real.
- Precisión medida: 87% en imágenes originales, 100% en sintéticas (datos de entrenamiento, optimista).
  No hay evaluación con cámara real. El paper menciona 94% y TFLite: no coincide con lo medido.
- El paper aún tiene párrafos de nota interna y ORCID "[completar]".
- Referencias (DS 328, Ley 1658, enlaces Minedu, alfabeto peruano) por verificar.
- Palabras en aimara/quechua: validar con un hablante.
- Solo reconoce 24 letras estáticas; CH, J, LL, Ñ, RR, Z (con movimiento) quedan como trabajo futuro.
