import sys
sys.path.insert(0,'tools')
import render_video as r, numpy as np
from PIL import Image
r.init()
for t in [float(x) for x in sys.argv[1:]]:
    fi=int(t*30)
    b=r.render_frame(fi)
    im=Image.frombytes('RGB',(r.W,r.H),b).resize((540,960),Image.LANCZOS)
    im.save(f'build/prev_{t:05.2f}.png')
