import cv2, numpy as np, glob, re, os, sys, random, zipfile
sys.path.insert(0,'.')
from key import key
rng=np.random.default_rng(7); random.seed(7)
H=W=224
OUT='gen_out'; os.makedirs(OUT,exist_ok=True)

# ---- reference skin chroma from the original cabinet photos
ref=[]
for f in sorted(glob.glob('tm/A-!-*.jpg'))[::20]:
    im=cv2.imread(f); y=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
    m=cv2.inRange(y,(40,138,80),(255,175,125))>0
    ref.append(cv2.cvtColor(im,cv2.COLOR_BGR2LAB).reshape(-1,3)[m.ravel()])
ref=np.concatenate(ref).astype(np.float32); RM,RS=ref.mean(0),ref.std(0)

def fix_color(im,m):
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32)
    px=lab[m>0]; mu,sd=px.mean(0),px.std(0)+1e-3
    out=lab.copy()
    for c in (1,2): out[...,c]=(lab[...,c]-mu[c])*(RS[c]/sd[c])*0.5+RM[c]
    return cv2.cvtColor(np.clip(out,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)

# ---- backgrounds
cab=cv2.imread('bg_plate.png')
room_fs=sorted(glob.glob('tm2/Sin_letra_o_vacio_2-!-*.jpg'))
room=[cv2.imread(f) for f in room_fs[::15]]
def lowfreq(sc=8,amp=18):
    n=rng.normal(0,1,(sc,sc,1)).astype(np.float32)
    return cv2.resize(n,(W,H),interpolation=cv2.INTER_CUBIC)[...,None]*amp
def synth():
    hue=rng.uniform(0,180)
    while 85<hue<135: hue=rng.uniform(0,180)
    sat=rng.uniform(0,90); val=rng.uniform(70,235)
    base=cv2.cvtColor(np.uint8([[[hue,sat,val]]]),cv2.COLOR_HSV2BGR)[0,0].astype(np.float32)
    img=np.tile(base,(H,W,1))
    ang=rng.uniform(0,2*np.pi); gx,gy=np.cos(ang),np.sin(ang)
    xs,ys=np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
    img+= (xs*gx+ys*gy)[...,None]*rng.uniform(-35,35)
    img+= lowfreq(rng.integers(3,10),rng.uniform(6,20))
    for _ in range(rng.integers(0,4)):          # soft rectangles: doors, frames, shelves
        x0,y0=rng.integers(0,W-40),rng.integers(0,H-40); x1,y1=x0+rng.integers(30,120),y0+rng.integers(30,150)
        col=rng.uniform(40,220,3).astype(np.float32); ov=img.copy()
        cv2.rectangle(ov,(x0,y0),(min(x1,W),min(y1,H)),col.tolist(),-1)
        img=cv2.addWeighted(img,0.6,cv2.GaussianBlur(ov,(9,9),0),0.4,0)
    img+=rng.normal(0,3,img.shape).astype(np.float32)
    return np.clip(img,0,255).astype(np.uint8)
def wood():
    base=np.array([rng.uniform(25,70),rng.uniform(45,95),rng.uniform(70,140)],np.float32)
    img=np.tile(base,(H,W,1))
    streak=cv2.resize(rng.normal(0,1,(1,40,1)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)[...,None]*14
    img+=streak+lowfreq(6,10)
    for _ in range(rng.integers(1,4)):
        x=rng.integers(0,W); cv2.line(img,(x,0),(x,H),(base*0.5).tolist(),rng.integers(1,3))
    return np.clip(img,0,255).astype(np.uint8)
def crop_zoom(src,zmax=1.8):
    z=rng.uniform(1.0,zmax); s=int(min(src.shape[:2])/z)
    y=rng.integers(0,src.shape[0]-s+1); x=rng.integers(0,src.shape[1]-s+1)
    c=src[y:y+s,x:x+s]
    if rng.random()<0.5: c=c[:,::-1]
    return cv2.resize(c,(W,H),interpolation=cv2.INTER_AREA)
def background():
    r=rng.random()
    if r<0.55: b=synth()
    elif r<0.70: b=crop_zoom(cab,1.5)
    elif r<0.85: b=crop_zoom(room[rng.integers(len(room))],2.2)
    else: b=wood()
    b=b.astype(np.float32)*rng.uniform(0.75,1.2)+rng.uniform(-12,12)
    return np.clip(b,0,255).astype(np.uint8)

# ---- compositing
def despill(im,edge):
    b,g,r=[im[...,i].astype(np.float32) for i in range(3)]
    lim=(g+r)/2; nb=np.where(edge>0,np.minimum(b,lim),b)
    out=im.copy(); out[...,0]=np.clip(nb,0,255).astype(np.uint8); return out
def compose(im):
    m=key(im)
    k=np.ones((3,3),np.uint8)
    m2=cv2.erode(m,k,iterations=1)
    edge=cv2.dilate(m,k,iterations=2)-cv2.erode(m,k,iterations=2)
    hand=despill(fix_color(im,m2),edge)
    flip=rng.random()<0.25
    if flip: hand=hand[:,::-1].copy(); m2=m2[:,::-1].copy()
    left=m2[:,:3].sum()>m2[:,-3:].sum()
    s=rng.uniform(0.85,1.1); ax=0 if left else W
    M=np.float32([[s,0,ax*(1-s)],[0,s,H*(1-s)]])
    hand=cv2.warpAffine(hand,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    m2=cv2.warpAffine(m2,M,(W,H),flags=cv2.INTER_LINEAR)
    a=(cv2.GaussianBlur(m2,(5,5),1.0).astype(np.float32)/255)[...,None]
    bg=background().astype(np.float32)
    out=hand.astype(np.float32)*a+bg*(1-a)
    out=out*rng.uniform(0.85,1.15)+rng.uniform(-10,10)
    out=cv2.GaussianBlur(out,(3,3),0.6) if rng.random()<0.3 else out
    return np.clip(out,0,255).astype(np.uint8)

def farthest(files,k):
    T=np.stack([cv2.resize(cv2.imread(f),(16,16),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255 for f in files])
    sel=[0]; d=((T-T[0])**2).mean(1)
    for _ in range(min(k,len(files))-1):
        i=int(d.argmax()); sel.append(i); d=np.minimum(d,((T-T[i])**2).mean(1))
    return [files[i] for i in sel]

if __name__=='__main__':
    PER=int(sys.argv[1]) if len(sys.argv)>1 else 110; REP=int(sys.argv[2]) if len(sys.argv)>2 else 2
    letters=[c for c in sorted(set(re.match(r'tm2/(.+)-!-\d+\.jpg',f).group(1) for f in glob.glob('tm2/*.jpg'))) if not c.startswith('Sin')]
    if len(sys.argv)>3: letters=sys.argv[3].split(',')
    for c in letters:
        fs=sorted(glob.glob(f'tm2/{c}-!-*.jpg'))
        sel=farthest(fs,PER)
        d=os.path.join(OUT,c); os.makedirs(d,exist_ok=True); n=0
        for f in sel:
            im=cv2.imread(f)
            for r in range(REP):
                cv2.imwrite(os.path.join(d,f'{c}_{n:04d}.jpg'),compose(im),[cv2.IMWRITE_JPEG_QUALITY,92]); n+=1
        print(c,n,flush=True)
