import cv2, numpy as np
def key(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[...,0].astype(int), hsv[...,1].astype(int), hsv[...,2].astype(int)
    m = (((h>=0)&(h<=45))|(h>=170)) & (s>25) & (v>70)
    m = m.astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    n,lab,st,_ = cv2.connectedComponentsWithStats(m)
    if n>1:
        k = 1+np.argmax(st[1:,cv2.CC_STAT_AREA]); m=((lab==k)*255).astype(np.uint8)
    # fill holes
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(m); cv2.drawContours(f, cnts, -1, 255, -1)
    return f
