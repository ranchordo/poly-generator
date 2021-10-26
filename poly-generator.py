import cv2
import numpy as np
import os

GRAD_DELTA=3
GRAD_RATE=1000*0.17

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
        self.msum=np.sum(self.mask)/255.
        self.mmsum=self.img.shape[0]*self.img.shape[1]
        self.center=self.getCenter()
        self.invalid=(xa<0) or (ya<0) or (xb<0) or (yb<0) or (xc<0) or (yc<0)
        if self.invalid:
            print("Recieved an invalid triangle. Ignoring...")
        self.mean=cv2.mean(self.img,mask=self.mask)
        #self.mean=self.img[int(self.center[1]),int(self.center[0])]
    def draw(self,drimg):
        if self.invalid: return
        cv2.drawContours(drimg,self.ctr,-1,(int(self.mean[0]),int(self.mean[1]),int(self.mean[2])),-1)
    def drawOutline(self,drimg):
        cv2.drawContours(drimg,self.ctr,-1,(0,150,255),3)
    def drawErrors(self,drimg):
        cv2.drawContours(drimg,self.ctr,-1,(self.getError(),0,0),-1)
    def getError(self):
        if self.invalid or self.msum<=1000:
            return -1
        mimg=np.full(self.img.shape,self.mean[0:3],np.uint8)
        dimg=cv2.absdiff(mimg, self.img)
        m=np.array(cv2.mean(dimg,mask=self.mask))
        return np.sqrt(np.abs(m.dot(m)))+(250*(self.msum/self.mmsum))
    def checkPointWithin(self,p):
        c1=(self.xb-self.xa)*(p[1]-self.ya)-(self.yb-self.ya)*(p[0]-self.xa)
        c2=(self.xc-self.xb)*(p[1]-self.yb)-(self.yc-self.yb)*(p[0]-self.xb)
        c3=(self.xa-self.xc)*(p[1]-self.yc)-(self.ya-self.yc)*(p[0]-self.xc)
        return (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0)
    def getCenter(self):
        mABx=(self.xa+self.xb)/2.0
        mABy=(self.ya+self.yb)/2.0
        mBCx=(self.xb+self.xc)/2.0
        mBCy=(self.yb+self.yc)/2.0

        mCmAB=(mABy-self.yc)/(mABx-self.xc)
        mAmBC=(mBCy-self.ya)/(mBCx-self.xa)
        if mCmAB==mAmBC:
            print("Evading a division by zero...")
            return (-1,-1)
        x=(-(mAmBC*self.xa)+(mCmAB*self.xc)+self.ya-self.yc)/(mCmAB-mAmBC)
        y=mCmAB*(x-self.xc)+self.yc
        return (x,y)
    def minimalErrorSubdivisionPoint(self):
        guess=self.center
        #return guess
        bestGuess=guess
        bestError=self.img.shape[0]*self.img.shape[1]*255
        iterations=0
        while True:
            guessx=(guess[0]+GRAD_DELTA,guess[1])
            guessy=(guess[0],guess[1]+GRAD_DELTA)
            err_base=self.getPhantomSubdivisionErrorMax(guess)
            err_x=self.getPhantomSubdivisionErrorMax(guessx)
            err_y=self.getPhantomSubdivisionErrorMax(guessy)
            dx=(err_x-err_base)/(GRAD_DELTA*1.)
            dy=(err_y-err_base)/(GRAD_DELTA*1.)
            guess=(guess[0]-(GRAD_RATE*dx),guess[1]-(GRAD_RATE*dy))
            if iterations>=5 or not self.checkPointWithin(guess):
                break;
            if guess[0]<0 or guess[0]>self.img.shape[0] or guess[1]<0 or guess[1]>self.img.shape[0]:
                break;
            if err_base<bestError:
                bestGuess=guess
                bestError=err_base
                iterations=0
            else:
                iterations+=1
            #print(dx,dy,guess,err_base,self.checkPointWithin(guess))
        #os._exit(0)
        return bestGuess
    def minimalErrorSubdivisionInfo(self):
        point=self.minimalErrorSubdivisionPoint()
        err_point=self.getPhantomSubdivisionErrorMax(point)
        err_cut=self.getPhantomSubdivisionErrorMax_cut()
        return (subdivide_point if err_point<err_cut else subdivide_cut),point
    def getPhantomSubdivisionErrorMax(self, p):
        nl=[]
        subdivide_point([self],0,p,nl=nl)
        errors=[tr.getError() for tr in nl]
        return np.max(errors) if len(errors)>0 else -1
    def getPhantomSubdivisionErrorMax_cut(self):
        nl=[]
        subdivide_cut([self],0,0,nl=nl)
        errors=[tr.getError() for tr in nl]
        return np.max(errors)*0.85 if len(errors)>0 else -1
def createTriangleContour(t):
    L=[[t.xa,t.ya],[t.xb,t.yb],[t.xc,t.yc]]
    return np.array(L).reshape((-1,1,2)).astype(np.int32)
def drawTris(ts,img):
    for t in ts:
        if t is not None:
            t.draw(img)
def append(nl,tri):
    if not tri.invalid:
        nl.append(tri)
def subdivide_cut(ts,i,p,nl=None):
    if nl is None:
        nl=ts
    mABx=(ts[i].xa+ts[i].xb)/2.0
    mABy=(ts[i].ya+ts[i].yb)/2.0
    mBCx=(ts[i].xb+ts[i].xc)/2.0
    mBCy=(ts[i].yb+ts[i].yc)/2.0
    mACx=(ts[i].xc+ts[i].xa)/2.0
    mACy=(ts[i].yc+ts[i].ya)/2.0
    append(nl,Triangle(ts[i].xa,ts[i].ya,mABx,mABy,mACx,mACy,ts[i].img))
    append(nl,Triangle(ts[i].xb,ts[i].yb,mABx,mABy,mBCx,mBCy,ts[i].img))
    append(nl,Triangle(ts[i].xc,ts[i].yc,mACx,mACy,mBCx,mBCy,ts[i].img))
    append(nl,Triangle(mABx,mABy,mACx,mACy,mBCx,mBCy,ts[i].img))
    ts[i]=None
def subdivide_point(ts, i, p, nl=None):
    #A, B, P
    #B, C, P
    #C, A, P
    if nl is None:
        nl=ts
    append(nl,Triangle(ts[i].xa,ts[i].ya,ts[i].xb,ts[i].yb,p[0],p[1],ts[i].img))
    append(nl,Triangle(ts[i].xb,ts[i].yb,ts[i].xc,ts[i].yc,p[0],p[1],ts[i].img))
    append(nl,Triangle(ts[i].xc,ts[i].yc,ts[i].xa,ts[i].ya,p[0],p[1],ts[i].img))
    ts[i]=None
    

img=cv2.imread(r'C:/Users/ranch/Downloads/dino.jpg')
img=cv2.GaussianBlur(img,(11,11),cv2.BORDER_DEFAULT)
#img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

tris=[Triangle(0,0,img.shape[1],0,0,img.shape[0],img),Triangle(img.shape[1],img.shape[0],img.shape[1],0,0,img.shape[0],img)]
drimg=np.zeros(img.shape,np.uint8)
blimg=None
alpha=0.93
while True:
    maxidx=-1
    err=-1
    for i in range(len(tris)):
        e=tris[i].getError()
        if e>err:
            err=e
            maxidx=i
    subdiv_func,subdiv=tris[maxidx].minimalErrorSubdivisionInfo()
    subdiv_func(tris,maxidx,subdiv)
    nl=[]
    for tr in tris:
        if tr is not None:
            if not tr.invalid:
                nl.append(tr)
    tris=nl
    drawTris(tris,drimg)
    tris[maxidx].drawOutline(drimg)
    blimg=cv2.addWeighted(drimg,alpha,img,1-alpha,0.0)
    cv2.imshow('tris',blimg)
    if cv2.waitKey(10)==27:
        cv2.destroyAllWindows()
        break
drawTris(tris,drimg)
blimg=cv2.addWeighted(drimg,alpha,img,1-alpha,0.0)
cv2.imshow('tris',blimg)
cv2.waitKey(50)
