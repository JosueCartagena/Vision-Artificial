import cv2, numpy as np, glob, re, os, sys, shutil
sys.path.insert(0,'.')
import gen
from gen import fix_color, despill, farthest, cab, room, rng, H, W
from key2 import key2 as key

# --- natural-looking backgrounds only (no coloured boxes)
def synth2():
    hue=rng.uniform(0,180)
    while 85<hue<135: hue=rng.uniform(0,180)
    sat=rng.uniform(0,70); val=rng.uniform(80,230)
    base=cv2.cvtColor(np.uint8([[[hue,sat,val]]]),cv2.COLOR_HSV2BGR)[0,0].astype(np.float32)
    img=np.tile(base,(H,W,1))
    ang=rng.uniform(0,2*np.pi); xs,ys=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
    img+=(xs*np.cos(ang)+ys*np.sin(ang))[...,None]*rng.uniform(-35,35)
    img+=gen.lowfreq(int(rng.integers(3,9)),rng.uniform(5,16))
    if rng.random()<0.35:   # door / frame edge: one vertical darker band with soft edge
        x0=int(rng.integers(20,W-60)); w=int(rng.integers(25,80))
        band=np.zeros((H,W,1),np.float32); band[:,x0:x0+w]=1; band=cv2.GaussianBlur(band,(0,0),3)[...,None] if band.ndim==2 else cv2.GaussianBlur(band,(0,0),3).reshape(H,W,1)
        img=img*(1-0.25*band)
    img+=rng.normal(0,3,img.shape).astype(np.float32)
    return np.clip(img,0,255).astype(np.uint8)
def background():
    r=rng.random()
    if r<0.62: b=synth2()
    elif r<0.84: b=gen.crop_zoom(cab,1.5)
    else: b=gen.wood()
    b=b.astype(np.float32)*rng.uniform(0.8,1.15)+rng.uniform(-10,10)
    return np.clip(b,0,255).astype(np.uint8)

def analyse(m):
    ys,xs=np.nonzero(m)
    if len(ys)<400: return None
    border=(ys>=H-3)|(xs<=2)|(xs>=W-3)
    if border.sum()<5: return None
    e=np.array([xs[border].mean(),ys[border].mean()])
    d=np.hypot(xs-e[0],ys-e[1]); far=d>0.62*d.max()
    fx,fy=xs[far],ys[far]
    return dict(x0=fx.min(),x1=fx.max(),y0=fy.min(),y1=fy.max(),dmax=d.max())

def valid_crop(m,box):
    x,y,S=box
    c=m[y:y+S,x:x+S]
    if c.sum()<400*255: return False
    if c[0].any(): return False                          # touches top edge
    side=lambda a:(a>0).mean()
    if side(c[:,0])>0.45 or side(c[:,-1])>0.45: return False   # arm only enters, never covers a whole side
    return True

ZOOM=(2.0,3.0)
def choose_crop(m,a):
    cx=(a['x0']+a['x1'])/2; cy=(a['y0']+a['y1'])/2
    bw=a['x1']-a['x0']+1; bh=a['y1']-a['y0']+1; size=max(bw,bh)
    for _ in range(14):
        S=int(np.clip(size*rng.uniform(*ZOOM),140,224))
        jx=rng.uniform(-0.07,0.07)*S; jy=rng.uniform(-0.05,0.05)*S
        x=int(round(cx+jx-0.5*S)); y=int(round(cy+jy-rng.uniform(0.40,0.55)*S))
        x=int(np.clip(x,0,W-S)); y=int(np.clip(y,0,H-S))
        # far (fingertip) region must sit well inside the crop
        mg=0.08*S
        if not (a['x0']-x>=mg and x+S-a['x1']>=mg and a['y0']-y>=mg): continue
        if not valid_crop(m,(x,y,S)): continue
        return x,y,S
    return None

def compose2(im):
    m=key(im); a=analyse(m)
    if a is None: return None
    box=choose_crop(m,a)
    if box is None: return None
    x,y,S=box
    k=np.ones((3,3),np.uint8)
    m2=cv2.erode(m,k,iterations=1)
    edge=cv2.dilate(m,k,iterations=2)-cv2.erode(m,k,iterations=2)
    hand=despill(fix_color(im,m2),edge)
    hand=cv2.resize(hand[y:y+S,x:x+S],(W,H),interpolation=cv2.INTER_CUBIC)
    mm=cv2.resize(m2[y:y+S,x:x+S],(W,H),interpolation=cv2.INTER_LINEAR)
    if rng.random()<0.25: hand=hand[:,::-1].copy(); mm=mm[:,::-1].copy()
    al=(cv2.GaussianBlur(mm,(5,5),1.0).astype(np.float32)/255)[...,None]
    bg=background().astype(np.float32)
    out=hand.astype(np.float32)*al+bg*(1-al)
    out=out*rng.uniform(0.88,1.12)+rng.uniform(-8,8)
    return np.clip(out,0,255).astype(np.uint8), mm

def final_check(img,mm):
    """post-check on the OUTPUT image: nothing cut at the top, fingertips inside, arm touches an edge"""
    if (mm[0]>0).any() or (mm[:, :1]>0).mean()>0.45 or (mm[:,-1:]>0).mean()>0.45: return False
    ys,xs=np.nonzero(mm>127)
    if len(ys)<0.04*H*W: return False
    if ys.min()<6: return False
    touch=max((mm[-1]>127).sum(),(mm[:,0]>127).sum(),(mm[:,-1]>127).sum())
    return bool(touch>=12)

if __name__=='__main__':
    PER=int(sys.argv[1]); REP=int(sys.argv[2]); only=sys.argv[3].split(',') if len(sys.argv)>3 else None
    OUT='gen2_out'; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT)
    letters=[c for c in sorted(set(re.match(r'tm2/(.+)-!-\d+\.jpg',f).group(1) for f in glob.glob('tm2/*.jpg'))) if not c.startswith('Sin')]
    if only: letters=only
    for c in letters:
        fs=sorted(glob.glob(f'tm2/{c}-!-*.jpg'))
        pool=farthest(fs,min(len(fs),PER*3))        # diverse candidates, extra to absorb rejects
        d=os.path.join(OUT,c); os.makedirs(d); n=0; used=0; rej=0
        for f in pool:
            if used>=PER: break
            im=cv2.imread(f); ok=0
            for r in range(REP):
                for _ in range(6):
                    res=compose2(im)
                    if res and final_check(*res):
                        cv2.imwrite(os.path.join(d,f'{c}_{n:04d}.jpg'),res[0],[cv2.IMWRITE_JPEG_QUALITY,92]); n+=1; ok+=1; break
            if ok: used+=1
            else: rej+=1
        print(c,'imgs',n,'sources used',used,'rejected sources',rej,flush=True)
