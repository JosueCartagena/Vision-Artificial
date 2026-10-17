"""Paso 2: interfaz. Reconoce el alfabeto LSB con la cámara y arma palabras."""
import tkinter as tk
from tkinter import ttk
import cv2, numpy as np
from PIL import Image, ImageTk
import tm_loader

UMBRAL_INICIAL = 0.80   # confianza mínima para aceptar una letra
FRAMES_ESTABLES = 12    # cuántos cuadros seguidos debe mantenerse la misma letra para escribirla (~0.5 s)

class App:
    def __init__(self, root):
        self.root = root; root.title("Alfabeto LSB - Reconocimiento")
        self.model, self.labels = tm_loader.load()
        self.cap = cv2.VideoCapture(0)
        self.texto = ""; self.ultima = None; self.cuenta = 0; self.bloqueada = None

        izq = ttk.Frame(root); izq.grid(row=0, column=0, padx=10, pady=10)
        der = ttk.Frame(root); der.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.video = ttk.Label(izq); self.video.pack()
        self.letra = tk.Label(der, text="-", font=("Helvetica", 96, "bold"), width=3); self.letra.pack()
        self.conf = ttk.Label(der, text="", font=("Helvetica", 14)); self.conf.pack()
        self.barras = ttk.Frame(der); self.barras.pack(pady=8, fill="x")
        self.top = []
        for _ in range(3):
            fila = ttk.Frame(self.barras); fila.pack(fill="x", pady=1)
            l = ttk.Label(fila, width=4, font=("Helvetica", 12, "bold")); l.pack(side="left")
            b = ttk.Progressbar(fila, length=170, maximum=100); b.pack(side="left")
            v = ttk.Label(fila, width=5); v.pack(side="left"); self.top.append((l, b, v))

        ttk.Label(der, text="Confianza mínima").pack(pady=(10, 0))
        self.umbral = tk.DoubleVar(value=UMBRAL_INICIAL)
        ttk.Scale(der, from_=0.5, to=0.99, variable=self.umbral, length=200).pack()

        ttk.Label(der, text="Texto formado:").pack(pady=(14, 0))
        self.salida = tk.Label(der, text="", font=("Helvetica", 22), bg="white", width=18, anchor="w", relief="sunken"); self.salida.pack(pady=4)
        bt = ttk.Frame(der); bt.pack()
        ttk.Button(bt, text="Espacio", command=lambda: self.add(" ")).pack(side="left", padx=2)
        ttk.Button(bt, text="Borrar", command=lambda: self.set(self.texto[:-1])).pack(side="left", padx=2)
        ttk.Button(bt, text="Limpiar", command=lambda: self.set("")).pack(side="left", padx=2)
        ttk.Label(der, text="Mantén la letra ~0.5 s para escribirla.\nBaja la mano o haz 'sin letra' entre letras.", foreground="gray").pack(pady=10)
        root.protocol("WM_DELETE_WINDOW", self.cerrar); self.loop()

    def add(self, c): self.set(self.texto + c)
    def set(self, t): self.texto = t; self.salida.config(text=t)

    def loop(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            nombre, conf, p = tm_loader.predict(self.model, self.labels, frame)
            vacio = nombre.startswith("Sin_letra")
            valida = (not vacio) and conf >= self.umbral.get()
            self.letra.config(text=nombre if valida else "-", fg="green" if valida else "gray")
            self.conf.config(text=f"{'(sin letra)' if vacio else nombre}  {conf*100:.0f}%")
            for (l, b, v), i in zip(self.top, np.argsort(p)[::-1][:3]):
                n = self.labels[i]; l.config(text="∅" if n.startswith("Sin_letra") else n); b["value"] = p[i]*100; v.config(text=f"{p[i]*100:.0f}%")
            # escritura estable: misma letra N cuadros seguidos, y no repetir hasta que cambie o se baje la mano
            if valida and nombre == self.ultima: self.cuenta += 1
            else: self.cuenta = 1 if valida else 0
            self.ultima = nombre if valida else None
            if not valida and (vacio or conf >= self.umbral.get()): self.bloqueada = None
            if valida and self.cuenta >= FRAMES_ESTABLES and nombre != self.bloqueada:
                self.add(nombre); self.bloqueada = nombre
            h, w = frame.shape[:2]; s = min(h, w); y, x = (h-s)//2, (w-s)//2
            cv2.rectangle(frame, (x, y), (x+s, y+s), (0, 255, 0), 2)
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (560, int(560*h/w)))
            self.tkimg = ImageTk.PhotoImage(Image.fromarray(img)); self.video.config(image=self.tkimg)
        self.root.after(30, self.loop)

    def cerrar(self): self.cap.release(); self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
