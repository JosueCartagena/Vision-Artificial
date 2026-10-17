import cv2, numpy as np
from key import key as key_skin
def key2(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    h,s,v=[hsv[...,i].astype(int) for i in range(3)]
    blue=((h>=85)&(h<=135)&(s>70))
    nb=(~blue)&(v>62)                      # drops the near-black sleeve
    nb=cv2.morphologyEx(nb.astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    sk=key_skin(im)
    zone=cv2.dilate(sk,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    m=((nb>0)&(zone>0)).astype(np.uint8)*255
    m=cv2.bitwise_or(m,sk)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    if n>1:
        k=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); m=((lab==k)*255).astype(np.uint8)
    # fill only holes that are NOT see-through background (keeps the gap of O, C, F, etc. open)
    cnts,hier=cv2.findContours(m,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    f=np.zeros_like(m)
    for i,c in enumerate(cnts):
        if hier[0][i][3]==-1: cv2.drawContours(f,[c],-1,255,-1)
    for i,c in enumerate(cnts):
        if hier[0][i][3]!=-1:
            hm=np.zeros_like(m); cv2.drawContours(hm,[c],-1,255,-1)
            hm=cv2.erode(hm,np.ones((3,3),np.uint8))
            tot=(hm>0).sum()
            if tot==0: cv2.drawContours(f,[c],-1,0,-1); continue
            bf=(blue&(hm>0)).sum()/tot
            if bf<0.25: pass            # not background: keep filled
            else: cv2.drawContours(f,[c],-1,0,-1)   # background visible through: open it
    f=cv2.GaussianBlur(f,(0,0),1.4); f=((f>127)*255).astype(np.uint8)
    return f
