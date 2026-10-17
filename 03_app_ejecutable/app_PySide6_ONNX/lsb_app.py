"""Reconocedor del alfabeto dactilológico LSB (Lengua de Señas Boliviana).
Modelo: Teachable Machine exportado a ONNX. Interfaz: PySide6. Cámara: OpenCV.

Uso:  python lsb_app.py            (cámara)
      python lsb_app.py --demo foto.jpg   (usa una foto en vez de la cámara, para probar la interfaz)
"""
import os, sys
import numpy as np, cv2
import onnxruntime as ort
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPainterPath, QPen
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton,
                               QSlider, QComboBox, QFrame, QCheckBox, QProgressBar, QGridLayout)

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
FRAMES_ESTABLES = 12      # cuadros seguidos con la misma letra para escribirla (~0,4 s)
VACIO = "Sin_letra_o_vacio"

QSS = """
* { font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial; color: #E8ECF4; }
QWidget#root { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0B1020, stop:1 #141B33); }
QFrame.card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; }
QLabel.muted { color: #8E99B5; font-size: 12px; }
QLabel.h { font-size: 11px; color: #8E99B5; letter-spacing: 2px; }
QPushButton { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 10px 16px; font-size: 14px; }
QPushButton:hover { background: rgba(255,255,255,0.15); }
QPushButton:pressed { background: rgba(255,255,255,0.22); }
QPushButton#accent { background: #3D7BFF; border: none; font-weight: 600; }
QPushButton#accent:hover { background: #5890FF; }
QPushButton#danger:hover { background: rgba(255,90,90,0.35); }
QComboBox { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 6px 10px; }
QComboBox QAbstractItemView { background: #1A2240; selection-background-color: #3D7BFF; }
QProgressBar { background: rgba(255,255,255,0.08); border: none; border-radius: 4px; max-height: 8px; min-height: 8px; }
QProgressBar::chunk { background: #3D7BFF; border-radius: 4px; }
QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,0.12); border-radius: 3px; }
QSlider::sub-page:horizontal { background: #3D7BFF; border-radius: 3px; }
QSlider::handle:horizontal { width: 18px; height: 18px; margin: -6px 0; border-radius: 9px; background: #FFFFFF; }
QCheckBox { font-size: 13px; }
"""


class Modelo:
    def __init__(self):
        self.sess = ort.InferenceSession(os.path.join(BASE, "model.onnx"), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.labels = [l.strip().split(" ", 1)[1] for l in open(os.path.join(BASE, "labels.txt"), encoding="utf-8") if l.strip()]

    def predecir(self, bgr):
        h, w = bgr.shape[:2]; s = min(h, w); y, x = (h - s) // 2, (w - s) // 2          # recorte central, como Teachable Machine
        im = cv2.resize(bgr[y:y + s, x:x + s], (224, 224), interpolation=cv2.INTER_AREA)
        im = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0)[None]
        return self.sess.run(None, {self.inp: im})[0][0]


class LetraCard(QWidget):
    """Tarjeta grande con la letra detectada y un anillo de confianza."""
    def __init__(self):
        super().__init__(); self.setMinimumSize(250, 250); self.letra = "·"; self.conf = 0.0; self.ok = False

    def set(self, letra, conf, ok): self.letra, self.conf, self.ok = letra, conf, ok; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        s = min(self.width(), self.height()) - 24; r = QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)
        p.setPen(QPen(QColor(255, 255, 255, 25), 12)); p.drawEllipse(r)
        col = QColor("#35D07F") if self.ok else QColor("#5B6684")
        pen = QPen(col, 12); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
        p.drawArc(r, 90 * 16, int(-360 * 16 * self.conf))
        p.setPen(col if self.ok else QColor("#8E99B5")); f = QFont("Segoe UI", int(s * 0.34), QFont.Bold); p.setFont(f)
        p.drawText(r.adjusted(0, -s * 0.06, 0, -s * 0.06), Qt.AlignCenter, self.letra)
        p.setPen(QColor("#8E99B5")); p.setFont(QFont("Segoe UI", 12))
        p.drawText(r.adjusted(0, s * 0.52, 0, 0), Qt.AlignHCenter | Qt.AlignTop, f"{self.conf * 100:.0f}% de confianza")


class Video(QLabel):
    def __init__(self):
        super().__init__(); self.setMinimumSize(640, 480); self.setAlignment(Qt.AlignCenter); self.pm = None

    def set_frame(self, rgb, ok):
        h, w, _ = rgb.shape; qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy(); self.pm = (QPixmap.fromImage(qi), ok); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath(); path.addRoundedRect(QRectF(self.rect()), 18, 18); p.setClipPath(path); p.fillRect(self.rect(), QColor("#0A0E1A"))
        if not self.pm:
            p.setPen(QColor("#8E99B5")); p.setFont(QFont("Segoe UI", 14)); p.drawText(self.rect(), Qt.AlignCenter, "Iniciando cámara…"); return
        pm, ok = self.pm; sc = pm.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        ox, oy = (sc.width() - self.width()) // 2, (sc.height() - self.height()) // 2; p.drawPixmap(-ox, -oy, sc)
        # cuadro guía = zona que ve el modelo (recorte central)
        s = min(self.width(), self.height()) - 36; r = QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)
        col = QColor("#35D07F") if ok else QColor(255, 255, 255, 170)
        p.setClipping(False); p.setPen(QPen(col, 3)); L = s * 0.12
        for (x, y, dx, dy) in [(r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1), (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)]:
            p.drawLine(int(x), int(y), int(x + dx * L), int(y)); p.drawLine(int(x), int(y), int(x), int(y + dy * L))


class App(QWidget):
    def __init__(self, demo=None):
        super().__init__(); self.setObjectName("root"); self.setWindowTitle("Alfabeto LSB · Reconocimiento con cámara"); self.resize(1180, 760)
        self.setStyleSheet(QSS); self.m = Modelo(); self.demo = cv2.imread(demo) if demo else None
        self.cap = None; self.texto = ""; self.ultima = None; self.cuenta = 0; self.bloq = None
        self.suave = np.zeros(len(self.m.labels), np.float32)

        root = QHBoxLayout(self); root.setContentsMargins(24, 22, 24, 22); root.setSpacing(20)
        izq = QVBoxLayout(); izq.setSpacing(14)
        t = QLabel("Alfabeto dactilológico LSB"); t.setStyleSheet("font-size:26px;font-weight:700;")
        st = QLabel("Reconocimiento de señas con cámara · Lengua de Señas Boliviana"); st.setProperty("class", "muted"); st.setStyleSheet("color:#8E99B5;font-size:13px;")
        izq.addWidget(t); izq.addWidget(st)
        self.video = Video(); izq.addWidget(self.video, 1)
        ctr = QHBoxLayout(); ctr.setSpacing(12)
        self.cam = QComboBox(); [self.cam.addItem(f"Cámara {i}", i) for i in range(4)]; self.cam.currentIndexChanged.connect(self.abrir)
        self.mir = QCheckBox("Espejo"); self.mir.setChecked(True)
        ctr.addWidget(self.cam); ctr.addWidget(self.mir); ctr.addStretch(); izq.addLayout(ctr)
        root.addLayout(izq, 3)

        der = QVBoxLayout(); der.setSpacing(14)
        c1 = self.card(); l1 = QVBoxLayout(c1); l1.addWidget(self.h("LETRA DETECTADA")); self.letra = LetraCard(); l1.addWidget(self.letra, 1); der.addWidget(c1, 3)
        c2 = self.card(); l2 = QVBoxLayout(c2); l2.addWidget(self.h("OTRAS POSIBILIDADES")); g = QGridLayout(); g.setHorizontalSpacing(10); self.top = []
        for i in range(3):
            a = QLabel("–"); a.setFixedWidth(34); a.setStyleSheet("font-size:16px;font-weight:700;"); b = QProgressBar(); b.setTextVisible(False); b.setMaximum(100)
            v = QLabel("0%"); v.setFixedWidth(42); v.setStyleSheet("color:#8E99B5;"); g.addWidget(a, i, 0); g.addWidget(b, i, 1); g.addWidget(v, i, 2); self.top.append((a, b, v))
        l2.addLayout(g); der.addWidget(c2)
        c3 = self.card(); l3 = QVBoxLayout(c3); l3.addWidget(self.h("TEXTO FORMADO"))
        self.salida = QLabel(""); self.salida.setMinimumHeight(54); self.salida.setWordWrap(True)
        self.salida.setStyleSheet("font-size:28px;font-weight:600;letter-spacing:3px;background:rgba(0,0,0,0.25);border-radius:12px;padding:6px 14px;")
        l3.addWidget(self.salida); bt = QHBoxLayout()
        for txt, fn, oid in [("Espacio", lambda: self.add(" "), "accent"), ("Borrar", lambda: self.set(self.texto[:-1]), ""), ("Limpiar", lambda: self.set(""), "danger")]:
            b = QPushButton(txt); b.setObjectName(oid); b.clicked.connect(fn); bt.addWidget(b)
        l3.addLayout(bt)
        fila = QHBoxLayout(); fila.addWidget(self.h("CONFIANZA MÍNIMA")); self.vu = QLabel("80%"); self.vu.setAlignment(Qt.AlignRight); fila.addWidget(self.vu); l3.addLayout(fila)
        self.sl = QSlider(Qt.Horizontal); self.sl.setRange(50, 99); self.sl.setValue(80); self.sl.valueChanged.connect(lambda v: self.vu.setText(f"{v}%")); l3.addWidget(self.sl)
        der.addWidget(c3)
        tip = QLabel("Mantén la seña quieta unos 0,4 s para escribirla. Baja la mano entre letras."); tip.setWordWrap(True); tip.setStyleSheet("color:#8E99B5;font-size:12px;"); der.addWidget(tip)
        root.addLayout(der, 2)

        self.abrir(); self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(30)

    def card(self): f = QFrame(); f.setProperty("class", "card"); f.setStyleSheet(""); f.setObjectName("c"); f.setStyleSheet("QFrame#c{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:18px;}"); return f
    def h(self, t): l = QLabel(t); l.setStyleSheet("font-size:11px;color:#8E99B5;letter-spacing:2px;font-weight:600;"); return l

    def abrir(self, *_):
        if self.demo is not None: return
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.cam.currentData() or 0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)

    def add(self, c): self.set(self.texto + c)
    def set(self, t): self.texto = t; self.salida.setText(t if t else "")

    def tick(self):
        if self.demo is not None: frame = self.demo.copy(); ok = True
        else: ok, frame = self.cap.read() if self.cap else (False, None)
        if not ok or frame is None: return
        if self.mir.isChecked(): frame = cv2.flip(frame, 1)
        p = self.m.predecir(frame); self.suave = 0.55 * self.suave + 0.45 * p if self.suave.any() else p      # suaviza para que no parpadee
        i = int(self.suave.argmax()); nombre = self.m.labels[i]; conf = float(self.suave[i]); vacio = nombre == VACIO
        valida = (not vacio) and conf >= self.sl.value() / 100
        self.letra.set("·" if not valida else nombre, conf if not vacio else 0.0, valida)
        for (a, b, v), j in zip(self.top, np.argsort(self.suave)[::-1][:3]):
            n = self.m.labels[j]; a.setText("∅" if n == VACIO else n); b.setValue(int(self.suave[j] * 100)); v.setText(f"{self.suave[j] * 100:.0f}%")
        self.cuenta = self.cuenta + 1 if (valida and nombre == self.ultima) else (1 if valida else 0); self.ultima = nombre if valida else None
        if not valida: self.bloq = None
        if valida and self.cuenta >= FRAMES_ESTABLES and nombre != self.bloq: self.add(nombre); self.bloq = nombre
        self.video.set_frame(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), valida)

    def closeEvent(self, e):
        if self.cap: self.cap.release()
        e.accept()


if __name__ == "__main__":
    demo = sys.argv[sys.argv.index("--demo") + 1] if "--demo" in sys.argv else None
    app = QApplication(sys.argv); w = App(demo); w.show()
    if "--shot" in sys.argv:                      # captura de pantalla para pruebas
        out = sys.argv[sys.argv.index("--shot") + 1]
        def shot():
            w.add("E"); w.add("A"); w.grab().save(out); app.quit()
        QTimer.singleShot(1500, shot)
    sys.exit(app.exec())
