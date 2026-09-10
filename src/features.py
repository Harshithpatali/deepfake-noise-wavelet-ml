import numpy as np, pywt
from PIL import Image,ImageOps

WAVELET="db2"; LEVEL=3
BANDS=["LL3","LH3","HL3","HH3","LH2","HL2","HH2","LH1","HL1","HH1"]
STATS=["mean","std","variance","energy","mad","skewness","kurtosis"]
NORMAL_COLS=["pixel_mean","pixel_std","pixel_variance","pixel_energy","pixel_mad","pixel_skewness","pixel_kurtosis",
"gradient_mean","gradient_std","gradient_energy","laplacian_mean","laplacian_std","laplacian_energy",
"noise_hf_mean","noise_hf_std","noise_hf_energy"]
WAVELET_COLS=[f"{b}_{s}" for b in BANDS for s in STATS]+["wavelet_sigma_hat"]

# Additional inexpensive forensic descriptors. All are derived from the same
# canonical grayscale image and the same single DWT used for the 71 wavelet features.
EXTRA_COLS=["noise_hf_median_abs","noise_hf_p95_abs","noise_hf_p99_abs",
"gradient_p50_abs","gradient_p90_abs","gradient_p95_abs","gradient_p99_abs",
"laplacian_p50_abs","laplacian_p90_abs","laplacian_p95_abs","laplacian_p99_abs",
"wavelet_detail_energy_ratio","wavelet_high_low_energy_ratio","wavelet_entropy",
"local_std_mean","local_std_std","local_std_p90","local_std_p95"]
ALL_COLS=NORMAL_COLS+WAVELET_COLS+EXTRA_COLS

def load_gray(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert("L").resize((256,256),Image.Resampling.LANCZOS)
        return np.asarray(im,dtype=np.float64)

def _moments(v):
    v=np.asarray(v,dtype=np.float64).ravel(); m=v.mean(); s=v.std()
    if s==0:return 0.,0.
    z=(v-m)/s
    return float(np.mean(z**3)),float(np.mean(z**4)-3)

def _stats(v):
    v=np.asarray(v,dtype=np.float64).ravel(); sk,ku=_moments(v); m=v.mean()
    return [m,v.std(),v.var(),np.mean(v*v),np.mean(np.abs(v-m)),sk,ku]

def normal_from_x(x):
    p=np.pad(x,1,mode="reflect")
    gx=np.diff(x,axis=1); gy=np.diff(x,axis=0); g=np.r_[gx.ravel(),gy.ravel()]
    lap=p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*x
    local=(p[:-2,:-2]+p[:-2,1:-1]+p[:-2,2:]+p[1:-1,:-2]+p[1:-1,1:-1]+p[1:-1,2:]+p[2:,:-2]+p[2:,1:-1]+p[2:,2:])/9
    hf=x-local
    return dict(zip(NORMAL_COLS,_stats(x)+[g.mean(),g.std(),np.mean(g*g),lap.mean(),lap.std(),np.mean(lap*lap),hf.mean(),hf.std(),np.mean(hf*hf)]))

def wavelet_from_x(x):
    c=pywt.wavedec2(x,WAVELET,level=LEVEL); out={}
    for b,v in zip(BANDS,[c[0],*c[1],*c[2],*c[3]]):
        out.update({f"{b}_{s}":float(a) for s,a in zip(STATS,_stats(v))})
    out["wavelet_sigma_hat"]=float(np.median(np.abs(c[3][2]))/0.6745)
    return out,c

def extra_from_x(x,c):
    p=np.pad(x,1,mode="reflect")
    local=(p[:-2,:-2]+p[:-2,1:-1]+p[:-2,2:]+p[1:-1,:-2]+p[1:-1,1:-1]+p[1:-1,2:]+p[2:,:-2]+p[2:,1:-1]+p[2:,2:])/9
    hf=x-local; g=np.r_[np.diff(x,axis=1).ravel(),np.diff(x,axis=0).ravel()]
    lap=p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*x
    m2=(p[:-2,:-2]**2+p[:-2,1:-1]**2+p[:-2,2:]**2+p[1:-1,:-2]**2+p[1:-1,1:-1]**2+p[1:-1,2:]**2+p[2:,:-2]**2+p[2:,1:-1]**2+p[2:,2:]**2)/9
    ls=np.sqrt(np.maximum(m2-local**2,0))
    details=np.concatenate([np.asarray(c[i][j]).ravel() for i in (1,2,3) for j in (0,1,2)])
    low=np.asarray(c[0]).ravel()
    hist=np.histogram(np.abs(details),bins=64)[0].astype(float); q=hist/(hist.sum()+1e-12)
    vals={
    "noise_hf_median_abs":np.median(np.abs(hf)),"noise_hf_p95_abs":np.quantile(np.abs(hf),.95),"noise_hf_p99_abs":np.quantile(np.abs(hf),.99),
    "gradient_p50_abs":np.quantile(np.abs(g),.5),"gradient_p90_abs":np.quantile(np.abs(g),.9),"gradient_p95_abs":np.quantile(np.abs(g),.95),"gradient_p99_abs":np.quantile(np.abs(g),.99),
    "laplacian_p50_abs":np.quantile(np.abs(lap),.5),"laplacian_p90_abs":np.quantile(np.abs(lap),.9),"laplacian_p95_abs":np.quantile(np.abs(lap),.95),"laplacian_p99_abs":np.quantile(np.abs(lap),.99),
    "wavelet_detail_energy_ratio":np.mean(details**2)/(np.mean(low**2)+1e-12),
    "wavelet_high_low_energy_ratio":np.mean(np.concatenate([np.asarray(c[i][2]).ravel() for i in (1,2,3)])**2)/(np.mean(low**2)+1e-12),
    "wavelet_entropy":-np.sum(q*np.log2(q+1e-12)),
    "local_std_mean":ls.mean(),"local_std_std":ls.std(),"local_std_p90":np.quantile(ls,.9),"local_std_p95":np.quantile(ls,.95)}
    return {k:float(v) for k,v in vals.items()}

def extract_all(path):
    x=load_gray(path); normal=normal_from_x(x); wave,c=wavelet_from_x(x); extra=extra_from_x(x,c)
    return {**normal,**wave,**extra}

def extract_all_fast(path):
    return extract_all(path)
