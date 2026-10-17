"""Paso 1: prueba rápida en consola/ventana. Salir con la tecla Q."""
import cv2, tm_loader
model, labels = tm_loader.load()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)                      # espejo, como en Teachable Machine
    letra, conf, p = tm_loader.predict(model, labels, frame)
    h, w = frame.shape[:2]; s = min(h, w); y, x = (h - s)//2, (w - s)//2
    cv2.rectangle(frame, (x, y), (x+s, y+s), (0, 255, 0), 2)   # zona que ve el modelo
    txt = "..." if letra.startswith("Sin_letra") else letra
    cv2.putText(frame, f"{txt} {conf*100:.0f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    cv2.imshow("Prueba LSB (Q para salir)", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")): break
cap.release(); cv2.destroyAllWindows()
