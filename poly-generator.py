import cv2
import numpy as np
import os

class Triangle:
    def __init__(self,xa,ya,xb,yb,xc,yc,img):
        self.xa=xa
        self.ya=ya
        self.xb=xb
        self.yb=yb
        self.xc=xc
        self.yc=yc
        self.img=img
        self.ctr=[np.array([[self.xa,self.ya],[self.xb,self.yb],[self.xc,self.yc]],dtype=np.int32)]
        self.mask=np.full(self.img.shape,0,np.uint8)
        cv2.drawContours(self.mask,self.ctr,-1,(255,255,255),-1)
        self.mask=cv2.cvtColor(self.mask,cv2.COLOR_BGR2GRAY)
        self.maskBGR=cv2.cvtColor(self.mask,cv2.COLOR_GRAY2BGR)
        self.msum=np.sum(self.mask)
        self.mean=cv2.mean(self.img,mask=self.mask)
    def draw(self,drimg):
        cv2.drawContours(drimg,self.ctr,-1,self.mean,-1)
    def drawOutline(self,drimg):
        cv2.drawContours(drimg,self.ctr,-1,(0,150,255),3)
    def drawErrors(self,drimg):
        cv2.drawContours(drimg,self.ctr,-1,(self.getError(),0,0),-1)
    def getError(self):
        mimg=np.full(self.img.shape,self.mean[0:3],np.int16)
        dimg=np.abs(self.img-mimg)
        dimg=dimg*self.maskBGR
        return np.sum(dimg)/self.msum
def createTriangleContour(t):
    L=[[t.xa,t.ya],[t.xb,t.yb],[t.xc,t.yc]]
    return np.array(L).reshape((-1,1,2)).astype(np.int32)
def drawTris(ts,img):
    for t in ts:
        if t is not None:
            t.drawErrors(img)
def subdivide(ts, i, p, nl=None):
    #A, B, P
    #B, C, P
    #C, A, P
    if nl is None:
        nl=ts
    nl.append(Triangle(ts[i].xa,ts[i].ya,ts[i].xb,ts[i].yb,p[0],p[1],ts[i].img))
    nl.append(Triangle(ts[i].xb,ts[i].yb,ts[i].xc,ts[i].yc,p[0],p[1],ts[i].img))
    nl.append(Triangle(ts[i].xc,ts[i].yc,ts[i].xa,ts[i].ya,p[0],p[1],ts[i].img))
    ts[i]=None
def getCenter(tri):
    mABx=(tri.xa+tri.xb)/2.0
    mABy=(tri.ya+tri.yb)/2.0
    mBCx=(tri.xb+tri.xc)/2.0
    mBCy=(tri.yb+tri.yc)/2.0

    mCmAB=(mABy-tri.yc)/(mABx-tri.xc)
    mAmBC=(mBCy-tri.ya)/(mBCx-tri.xa)
    #print(mCmAB,mAmBC)
    x=(-(mAmBC*tri.xa)+(mCmAB*tri.xc)+tri.ya-tri.yc)/(mCmAB-mAmBC)
    y=mCmAB*(x-tri.xc)+tri.yc
    #print(x,y)
    return x,y
    

num_tris=200
img=cv2.imread(r'/home/ranchordo/Desktop/keyboard.jpg')
#img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

tris=[Triangle(0,0,img.shape[0],0,0,img.shape[1],img),Triangle(img.shape[0],img.shape[1],img.shape[0],0,0,img.shape[1],img)]
drimg=np.zeros(img.shape,np.uint8)

for i in range(1):
    nl=[]
    for i in range(len(tris)):
        subdivide(tris,i,getCenter(tris[i]),nl)
    tris=nl

drawTris(tris,drimg)

#img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
#drimg=cv2.cvtColor(drimg,cv2.COLOR_HSV2BGR)
cv2.imshow('tris',drimg)
cv2.waitKey(0)
os._exit(0)
