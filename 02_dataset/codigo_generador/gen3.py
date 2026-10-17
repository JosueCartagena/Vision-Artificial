import cv2, numpy as np, glob, re, os, sys, shutil, zipfile, collections
sys.path.insert(0,'.')
import gen
from gen import fix_color, despill, farthest, cab, rng, H, W
from key2 import key2 as key
import gen2

# ---------- wall-like backgrounds by type (plaster texture, soft light, soft shadows; no stripes)
def wall(base,grain=1.8,blotch=5.0,grad=12.0,vig=0.10):
    img=np.tile(np.array(base,np.float32),(H,W,1))
    xs,ys=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
    ang=rng.uniform(0,2*np.pi)
    img+=(xs*np.cos(ang)+ys*np.sin(ang))[...,None]*rng.uniform(-grad,grad)           # light falling across the wall
    img+=gen.lowfreq(int(rng.integers(2,5)),blotch)                                     # large soft patches of paint / light
    img+=gen.lowfreq(int(rng.integers(10,22)),blotch*0.35)                              # plaster mottling
    r=np.sqrt(xs**2+ys**2)[...,None]; cx,cy=rng.uniform(-.6,.6,2)
    img*=1-vig*np.clip(np.sqrt((xs-cx)**2+(ys-cy)**2)[...,None],0,1.5)                   # darker toward the corners
    img+=rng.normal(0,grain,img.shape).astype(np.float32)
    return np.clip(img,0,255).astype(np.uint8)
def bgr_from_hsv(h,s,v): return cv2.cvtColor(np.uint8([[[h,s,v]]]),cv2.COLOR_HSV2BGR)[0,0].astype(np.float32)
def bg_white():
    r=rng.random()
    if r<0.55: base=bgr_from_hsv(rng.uniform(10,30),rng.uniform(0,10),rng.uniform(232,252))      # white / warm white
    elif r<0.8: base=bgr_from_hsv(rng.uniform(0,180),rng.uniform(0,6),rng.uniform(205,232))      # light grey
    else:
        h=rng.uniform(0,180)
        while 90<h<130: h=rng.uniform(0,180)
        base=bgr_from_hsv(h,rng.uniform(10,32),rng.uniform(205,240))                              # pale tinted wall (cream, mint, pink)
    return wall(base,grain=1.5,blotch=4.0,grad=10,vig=0.06)
def bg_black():
    base=bgr_from_hsv(rng.uniform(0,180),rng.uniform(0,40),rng.uniform(10,44))
    return wall(base,grain=2.0,blotch=3.0,grad=6,vig=0.18)
def bg_brown():
    if rng.random()<0.5:
        b=gen.crop_zoom(cab,1.7).astype(np.float32)*rng.uniform(0.8,1.15)+rng.uniform(-8,8)
        return np.clip(b,0,255).astype(np.uint8)
    base=bgr_from_hsv(rng.uniform(8,22),rng.uniform(70,160),rng.uniform(60,150))                # brown / tan / caramel painted wall
    return wall(base,grain=2.0,blotch=6.0,grad=14,vig=0.12)
BG={'blanco':bg_white,'cafe':bg_brown,'negro':bg_black}

# ---------- composition with near/far variation
MIN_PALM=0.27
_clahe=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(4,4))
def enhance(img,mask):
    """more local contrast + a little sharpening, only on the hand (finger detail), no halo at the outline"""
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); L=lab[...,0]
    lab[...,0]=cv2.addWeighted(L,0.45,_clahe.apply(L),0.55,0)
    e=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR).astype(np.float32)
    bl=cv2.GaussianBlur(e,(0,0),1.1); e=np.clip(e+0.75*(e-bl),0,255)
    inner=cv2.GaussianBlur(cv2.erode((mask>127).astype(np.uint8)*255,np.ones((5,5),np.uint8)),(0,0),1.5).astype(np.float32)[...,None]/255
    return np.clip(img.astype(np.float32)*(1-inner)+e*inner,0,255).astype(np.uint8)
def palm(m):
    dt=cv2.distanceTransform((m>127).astype(np.uint8)*255,cv2.DIST_L2,5)
    return 2*dt.max()/224

def compose3(im,bgtype):
    m=key(im); a=gen2.analyse(m)
    if a is None: return None
    box=gen2.choose_crop(m,a)
    if box is None: return None
    x,y,S=box
    k=np.ones((3,3),np.uint8)
    m2=cv2.erode(m,k,iterations=1)
    edge=cv2.dilate(m,k,iterations=2)-cv2.erode(m,k,iterations=2)
    hand=despill(fix_color(im,m2),edge)
    hand=cv2.resize(hand[y:y+S,x:x+S],(W,H),interpolation=cv2.INTER_LANCZOS4)
    mm=cv2.resize(m2[y:y+S,x:x+S],(W,H),interpolation=cv2.INTER_LINEAR)
    hand=enhance(hand,mm)
    # far shots: shrink the hand around the point where the arm enters the frame (arm keeps touching the edge)
    pw=palm(mm); smin=max(MIN_PALM/max(pw,1e-3),0.72)
    if smin<0.97 and rng.random()<0.5:
        s=rng.uniform(smin,0.97)
        ys_,xs_=np.nonzero(mm>127)
        nb_=(ys_>=H-3).sum(); nl_=(xs_<=2).sum(); nr_=(xs_>=W-3).sum()
        if nb_+nl_+nr_>=5:
            if nb_>=5 and (nl_>=5 or nr_>=5): ex,ey=(0 if nl_>=nr_ else W),H           # arm enters at a corner
            elif nb_>=5: ex,ey=xs_[ys_>=H-3].mean(),H
            elif nl_>=nr_: ex,ey=0,ys_[xs_<=2].mean()
            else: ex,ey=W,ys_[xs_>=W-3].mean()
            M=np.float32([[s,0,ex*(1-s)],[0,s,ey*(1-s)]])
            hand=cv2.warpAffine(hand,M,(W,H),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)
            mm=cv2.warpAffine(mm,M,(W,H),flags=cv2.INTER_AREA)
    if rng.random()<0.25: hand=hand[:,::-1].copy(); mm=mm[:,::-1].copy()
    al=(cv2.GaussianBlur(mm,(5,5),1.0).astype(np.float32)/255)[...,None]
    bg=BG[bgtype]().astype(np.float32)
    out=hand.astype(np.float32)*al+bg*(1-al)
    jit=rng.uniform(0.9,1.1) if bgtype!='negro' else rng.uniform(0.95,1.1)
    out=out*jit
    return np.clip(out,0,255).astype(np.uint8), mm

def check3(mm):
    if (mm[0]>0).any(): return False
    if (mm[:,0]>127).mean()>0.45 or (mm[:,-1]>127).mean()>0.45: return False
    ys,xs=np.nonzero(mm>127)
    if len(ys)<0.025*H*W or ys.min()<6: return False
    touch=max((mm[-1]>127).sum(),(mm[:,0]>127).sum(),(mm[:,-1]>127).sum())
    return touch>=8 and palm(mm)>=MIN_PALM-0.01

def blue_keep(im):
    """real blue-screen photo kept as is; only the hand's yellow tint is corrected"""
    m=key(im)
    corr=fix_color(im,cv2.erode(m,np.ones((3,3),np.uint8)))
    a=(cv2.GaussianBlur(m,(0,0),1.2).astype(np.float32)/255)[...,None]
    out=np.clip(corr*a+im*(1-a),0,255).astype(np.uint8)
    return enhance(out,m)

if __name__=='__main__':
    NBLUE=int(sys.argv[1]); NEACH=int(sys.argv[2]); only=sys.argv[3].split(',') if len(sys.argv)>3 else None
    OUT='gen3_out'; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
    letters=[c for c in sorted(set(re.match(r'tm2/(.+)-!-\d+\.jpg',f).group(1) for f in glob.glob('tm2/*.jpg'))) if not c.startswith('Sin')]
    if only: letters=only
    HARD={'K','M','N','O','C','P','T'}
    for c in letters:
        global_cfg=(60,80,0.36) if c in HARD|{'E','A','S'} else (60,80,0.30)      # 300 per letter, balanced; look-alikes get larger hands
        gen2.ZOOM=(1.7,2.4) if c in HARD|{'E','A','S'} else (2.0,3.0)
        BLUR=20 if c in HARD|{'E','A','S'} else 30      # blue photos, per-background photos, min hand size
        NB_,NE_,MIN_PALM=global_cfg
        fs=sorted(glob.glob(f'tm2/{c}-!-*.jpg'))
        d=os.path.join(OUT,c); os.makedirs(d); n=0
        gr=[cv2.Laplacian(cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() for f in fs]
        thr=np.percentile(gr,BLUR); fs_ok=[f for f,g in zip(fs,gr) if g>=thr]      # drop the blurriest 30%
        fs_ok=[f for f in fs_ok if palm(key(cv2.imread(f)))>=MIN_PALM]
        print(c,'sources medium/large:',len(fs_ok),flush=True)
        pool=farthest(fs_ok,len(fs_ok))
        for f in pool[:NB_]:
            cv2.imwrite(f'{d}/{c}_azul_{n:04d}.jpg',blue_keep(cv2.imread(f)),[cv2.IMWRITE_JPEG_QUALITY,92]); n+=1
        cnt=collections.Counter(); tries=0; i=0
        types=list(BG)
        while min(cnt[t] for t in types)<NE_ and tries<9000:
            t=min(types,key=lambda k:cnt[k])
            if cnt[t]>=NE_: t=[k for k in types if cnt[k]<NE_][0]
            f=pool[i%len(pool)]; i+=1; tries+=1
            res=compose3(cv2.imread(f),t)
            if res and check3(res[1]):
                cv2.imwrite(f'{d}/{c}_{t}_{n:04d}.jpg',res[0],[cv2.IMWRITE_JPEG_QUALITY,92]); n+=1; cnt[t]+=1
        print(c,n,dict(cnt),'tries',tries,flush=True)
