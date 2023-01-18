import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def color_space_demo():
    image=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    cv.imshow("lena",image)
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    ycrcb=cv.cvtColor(image,cv.COLOR_BGR2YCrCb)
    cv.imshow("hsv",hsv)
    cv.imshow("ycrcb",ycrcb)
    cv.waitKey(0)
    cv.destroyAllWindows()


def numpy_demo():
    m1=np.array([[2,3],[4,5]],dtype=np.uint8)
    # print(m1)
    m2=np.zeros((512,512,3),dtype=np.uint8)
    m2[0:256,:256]=(255,0,0)
    m2[0:256,256:]=(0,255,0)
    m2[256:,:256]=(0,0,255)
    m2[256:,256:]=255

    cv.imshow("m2",m2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # print(m2)
    m2[:]=(255,0,0)
    # print(m2)
    m3=np.zeros((4,4,3),dtype=np.uint8)
    # print(m3)
    # m3[:]=255
    # print(m3)


def visit_pixel_demo():
    image=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    cv.imshow("lena",image)
    h,w,c=image.shape
    for row in range(0,w,1):
        for col in range(0,w,1):
            b,g,r=image[row,col]
            image[row,col]=(255-b,255-g,255-r)
    cv.imshow("visited",image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def arithmetic_demo():
    image1=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    image2=np.zeros_like(image1)
    image2[:,:]=(110,0,250)
    cv.imshow("image1",image1)
    cv.imshow("image2",image2)
    added=cv.add(image1,image2)
    cv.imshow("added",added)
    h,w,c=image1.shape
    mask=np.zeros((h,w),dtype=np.uint8)
    mask[200:400,200:400]=1
    dst=cv.add(image1,image2,mask=mask)
    cv.imshow("dst",dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def trackbar_callback(pos):
    print(pos)

def trackbar_demo():
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    cv.namedWindow("trackbar_demo",cv.WINDOW_KEEPRATIO)
    cv.createTrackbar("lightness","trackbar_demo",0,200,trackbar_callback)
    cv.imshow("trackbar_demo",image)
    while True:
        pos=cv.getTrackbarPos("lightness","trackbar_demo")
        image2=np.zeros_like(image)
        image2[:,:]=(np.uint8(pos),np.uint8(pos),np.uint8(pos))
        #提升亮度
        result=cv.add(image,image2)
        #降低亮度
        #result=cv.subtract(image,image2)
        cv.imshow("trackbar_demo",result)
        c=cv.waitKey(1)
        if c==27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


def keyboard_demo():
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.namedWindow("keyboard_demo",cv.WINDOW_AUTOSIZE)
    cv.imshow("keyboard_demo",image)
    while True:
        c=cv.waitKey(10)
        #ESC
        if c == 27:
            break
        #Key0
        elif c ==48:
            cv.imshow("keyboard_demo",image)
        #key1
        elif c == 49:
            hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
            cv.imshow("keyboard_demo",hsv)
        #Key2
        elif c == 50:
            ycrcb=cv.cvtColor(image,cv.COLOR_BGR2YCrCb)
            cv.imshow("keyboard_demo",ycrcb)
        #Key3
        elif c == 51:
            rgb=cv.cvtColor(image,cv.COLOR_BGR2RGB)
            cv.imshow("keyboard_demo",rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()


def lut_demo():
    cv.namedWindow("lut_demo",cv.WINDOW_NORMAL)
    lut=[[255,0,255],[125,0,0],[127,255,200],[200,127,127],[0,255,255]]
    m1=np.array([[2,1,3,0],[2,2,1,1],[3,3,4,4],[4,4,1,1]])
    m2=np.zeros((4,4,3),dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            index=m1[i,j]
            m2[i,j]=lut[index]
    
    cv.imshow("lut_demo",m2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #建立查找表
    lut2 = np.zeros((256),dtype=np.uint8)
    gamma=0.7
    for i in range(256):
        print(i,"--",np.log(i/255.0))
        lut2[i]=int(np.exp(np.log(i/255.0)*gamma)*255.0)
    print(lut2)
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.namedWindow("butterfly_gamma",cv.WINDOW_AUTOSIZE)
    h,w,c=image.shape
    for row in range(h):
        for col in range(w):
            b,g,r=image[row,col]
            image[row,col]=(lut2[b],lut2[g],lut2[r])
    cv.imshow("buttefly_gamma",image)

    #自定义查找表
    lut3=np.zeros((256,1,3),dtype=np.uint8)
    for i in range(256):
        print(i,"--",np.log(i/255.0))
        c=int(np.exp(np.log(i/255.0)*gamma)*255.0)
        lut3[i,0]=(c,c,c)
    print(lut3)
    dst=cv.LUT(image,lut3)
    cv.imshow("butterfly_gamma",dst)

    #系统查找表
    dst=cv.applyColorMap(image,cv.COLORMAP_PINK)
    cv.imshow("butterfly_pink",dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


def task_3():
    task=np.array([cv.COLORMAP_AUTUMN,cv.COLORMAP_JET,cv.COLORMAP_RAINBOW,cv.COLORMAP_OCEAN,cv.COLORMAP_SUMMER,cv.COLORMAP_SPRING,cv.COLORMAP_COOL,cv.COLORMAP_HSV,cv.COLORMAP_PINK,cv.COLORMAP_HOT,cv.COLORMAP_PARULA,cv.COLORMAP_MAGMA,cv.COLORMAP_INFERNO,cv.COLORMAP_PLASMA,cv.COLORMAP_VIRIDIS,cv.COLORMAP_TWILIGHT,cv.COLORMAP_TWILIGHT_SHIFTED,cv.COLORMAP_TURBO,cv.COLORMAP_DEEPGREEN])
    cv.namedWindow("task_3",cv.WINDOW_KEEPRATIO)
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.createTrackbar("COLORMAP","task_3",0,19,trackbar_callback)
    cv.imshow("task_3",image)
    while True:
        pos=cv.getTrackbarPos("COLORMAP","task_3")
        dst=cv.applyColorMap(image,task[pos-1])
        cv.imshow("task_3",dst)
        c=cv.waitKey(1)
        if c==27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


def channel_splits():
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.namedWindow("butterfly",cv.WINDOW_AUTOSIZE)
    cv.imshow("butterfly",image)
    mv=cv.split(image)
    dst=np.zeros_like(image)
    #BGR2GRB
    cv.mixChannels([image],[dst],fromTo=[0,2,1,1,2,0])
    cv.imshow("mix_channels",dst)
    mask=cv.inRange(image,(43,46,100),(128,200,200))
    cv.imshow("inRange",mask)
    cv.waitKey(0)
    cv.destroyAllWindows()


def stats_demo():
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.imshow("butterfly",image)
    bgr_m=cv.mean(image)
    sub_m=np.float32(image)[:,:] -(bgr_m[0],bgr_m[1],bgr_m[2])
    result=sub_m*0.5
    result=result[:,:]+(bgr_m[0],bgr_m[1],bgr_m[2])
    cv.imshow("low-contrast-butterfly",cv.convertScaleAbs(result))

    result=sub_m*2.0
    result=result[:,:]+(bgr_m[0],bgr_m[1],bgr_m[2])
    cv.imshow("high-contrast-butterfly",cv.convertScaleAbs(result))
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_demo():
    canvas=np.zeros((512,512,3),dtype=np.uint8)

    # cv.rectangle(canvas,(100,100),(300,300),(0,0,255),2,8)
    # cv.circle(canvas,(250,250),50,(255,0,0),4,cv.LINE_8)
    # cv.line(canvas,(100,100),(300,300),(0,255,0),2,8)
    # cv.putText(canvas,"OpenCV-Python",(100,100),cv.FONT_HERSHEY_SIMPLEX,1.0,(255,0,255))
    # cv.imshow("canvas",canvas)
    # cv.waitKey(0)

    # 动态合理显示文本区域
    font_color=(140,199,0)
    cv.rectangle(canvas,(100,100),(300,300),font_color,2,8)
    label_txt="OpenCV-Python"
    font=cv.FONT_HERSHEY_SIMPLEX
    font_scale=0.5
    thickness=1
    (fw,uph),dh=cv.getTextSize(label_txt,font,font_scale,thickness)
    cv.rectangle(canvas,(100,100-uph-dh),(100+fw,100),(255,255,255),-1,8)
    cv.putText(canvas,label_txt,(100,100-dh),font,font_scale,(255,0,255),thickness)
    cv.imshow("canvas",canvas)
    cv.waitKey(0)

    cv.waitKey(0)
    cv.destroyAllWindows()

def random_demo():
    canvas=np.zeros((512,512,3),dtype=np.uint8)
    # random draw
    while True:
        b,g,r=np.random.randint(0,256,size=3)
        x1=np.random.randint(0,512)
        x2=np.random.randint(0,512)
        y1=np.random.randint(0,512)
        y2=np.random.randint(0,512)
        cv.rectangle(canvas,(x1,y1),(x2,y2),(int(b),int(g),int(r)),-1,8)
        cv.imshow("canvas",canvas)
        c=cv.waitKey(50)
        if c==27:
            break

        # reset background 每一次都会清除已随机生成的矩形,将其设置为黑色，没有这行代码则会持续生成，不会清除
        #cv.rectangle(canvas,(0,0),(512,512),(0,0,0),-1,8)

    cv.randn(canvas,(120,100,140),(30,50,20))
    cv.imshow("noise image",canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()

def poly_demo():
    canvas=np.zeros((512,512,3),dtype=np.uint8)
    pts=[]
    pts.append((100,100))
    pts.append((200,50))
    pts.append((280,100))
    pts.append((290,300))
    pts.append((50,300))
    pts=np.asarray(pts,dtype=np.int32)
    print(pts.shape)

    pts2=[]
    pts2.append((300,300))
    pts2.append((400,250))
    pts2.append((500,300))
    pts2.append((500,500))
    pts2.append((250,500))
    pts2=np.asarray(pts2,dtype=np.int32)
    print(pts2.shape)

    cv.polylines(canvas,[pts,pts2],True,(0,255,255),2,8)
    cv.fillPoly(canvas,[pts,pts2],(0,0,255),8,0)
    cv.imshow("poly_demo",canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


b1=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
img=np.copy(b1)
x1=-1
y1=-1
x2=-1
y2=-2



def mouse_drawing(event,x,y,flags,param):
    global x1,x2,y1,y2
    if event==cv.EVENT_LBUTTONDOWN:
        x1=x
        y1=y
    if event==cv.EVENT_MOUSEMOVE:
        if x1<0 or y1 <0:
            return
        x2=x
        y2=y
        dx=x2-x1
        dy=y2-y1
        #if dx>0 and dy>0:
            # 清除每次画的图形
        b1[:,:,:]=img[:,:,:]
        cv.rectangle(b1,(x1,y1),(x2,y2),(0,0,255),2,8,0)
            #cv.circle(b1,(x1,y1),int((x**2+y**2)**0.5),(0,0,255),4,cv.LINE_8)
        #cv.line(b1,(x1,y1),(x2,y2),(0,0,255),2,8,0)
    if event==cv.EVENT_LBUTTONUP:
        x2=x
        y2=y
        dx=x2-x1
        dy=y2-y1
        #if dx>0 and dy>0:
            # 清除每次画的图形
            #b1[:,:,:]=img[:,:,:]
        cv.rectangle(b1,(x1,y1),(x2,y2),(0,0,255),2,8,0)
            #cv.circle(b1,(x1,y1),int((x**2+y**2)**0.5),(0,0,255),4,cv.LINE_8)
        # 为下一次绘制做好准备
        #cv.line(b1,(x1,y1),(x2,y2),(0,0,255),2,8,0)
        x1=-1
        x2=-1
        y1=-1
        y2=-1

def mouse_demo():
    cv.namedWindow("mouse_demo",cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("mouse_demo",mouse_drawing)
    while True:
        cv.imshow("mouse_demo",b1)
        c=cv.waitKey(10)
        if c == 27:
            break
    cv.destroyAllWindows()
    

def norm_demo():
    image_uint8=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.imshow("image_uint8",image_uint8)
    img_f32=np.float32(image_uint8)
    cv.imshow("img_f32",img_f32)
    cv.normalize(img_f32,img_f32,1,0,cv.NORM_MINMAX)
    cv.imshow("norm_imgf32",img_f32)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.namedWindow("norm_demo",cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("normtype","norm_demo",0,3,trackbar_callback)
    while True:
        gray=cv.cvtColor(image_uint8,cv.COLOR_RGB2GRAY)
        dst=np.float32(gray)
        pos=cv.getTrackbarPos("normtype","norm_demo")
        if pos == 0:
            cv.normalize(dst,dst,1,0,cv.NORM_MINMAX)
        if pos == 1:
            cv.normalize(dst,dst,1,0,cv.NORM_L1)
        if pos == 2:            
            cv.normalize(dst,dst,1,0,cv.NORM_L2)            
        if pos == 3:
            cv.normalize(dst,dst,1,0,cv.NORM_INF)        
        cv.imshow("norm_demo",dst)
        c=cv.waitKey(50)
        if c ==27:
            break
    cv.destroyAllWindows()


def affine_demo():
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    h,w,c=image.shape
    cx=int(w/2)
    cy=int(h/2)
    cv.imshow("image",image)

    M=np.zeros((2,3),dtype=np.float32)
    M[0,0]=.7
    M[1,1]=.7
    M[0,2]=0
    M[1,2]=0
    print("(M(2×3)=\n",M)
    dst=cv.warpAffine(image,M,(int(w*.7),int(h*.7)))
    cv.imshow("rescale_demo",dst)
    
    # 获取旋转矩阵，degree>0表示逆时针旋转，原点在左上角
    M=cv.getRotationMatrix2D((w/2,h/2),45.0,1.0)
    dst=cv.warpAffine(image,M,(w,h))
    cv.imshow("rotate_demo",dst)

    dst=cv.flip(image,1) # 第二个参数等零表示上下翻转，等1表示左右翻转
    cv.imshow("flip_demo",dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


def video_demo():
    cap=cv.VideoCapture("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//vtest.avi")
    # query video file metadata
    fps=cap.get(cv.CAP_PROP_FPS)
    frame_w=cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_h=cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(fps,frame_w,frame_h)
    #encode mode
    #fourcc=cv.VideoWriter_fourcc(*"vp09")
    fourcc=cap.get(cv.CAP_PROP_FOURCC) # 编码格式
    #create Video writer
    writer=cv.VideoWriter("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//output.mp4",int(fourcc),fps,(int(frame_w),int(frame_h)))
    while True:
        ret,frame=cap.read()
        if ret is not True:
            break
        writer.write(frame)
        cv.imshow("frame",frame)
        c=cv.waitKey(1)
        if c == 27:
            break

    cap.release()
    writer.release()

    cv.waitKey(0)
    cv.destroyAllWindows()

# xwh NLP作业
"""def sum():
    sum11=np.zeros((512,512,3),dtype=np.uint8)
    sum=0
    for i in range(1,101):
        sum+=i
    label_txt="1+2+3+...+={}".format(sum)
    cv.putText(sum11,label_txt,(100,256),cv.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255))
    cv.imshow("sum11",sum11)
    cv.waitKey(0)
    cv.destroyAllWindows()"""

def image_hist():
    image=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
    cv.imshow("input",image)
    color =('blue','green','red')
    for i,color in enumerate(color):
        hist=cv.calcHist([image],[i],None,[32],[0,256])
        print(hist.dtype)
        plt.plot(hist,color=color)
        plt.xlim([0,32])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


def eq_demo():
    image=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//lena.jpg",cv.IMREAD_GRAYSCALE)
    cv.imshow("input",image)
    hist=cv.calcHist([image],[0],None,[32],[0,256])
    print(hist.dtype)
    plt.plot(hist,color='gray')
    plt.xlim([0,32])
    plt.show()
    cv.waitKey(0)

    eqimg=cv.equalizeHist(image)
    cv.imshow("eq",eqimg)
    hist=cv.calcHist([eqimg],[0],None,[32],[0,256])
    print(hist.dtype)
    plt.plot(hist,color='gray')
    plt.xlim([0,32])
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()


def conv_demo():
    image=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    dst=np.copy(image)
    cv.imshow("input",image)
    h,w,c=image.shape
    # 手动实现卷积
    for row in range(1,h-1,1):
        for col in range(1,w-1,1):
            m=cv.mean(image[row-2:row+2,col-2:col+2])
            dst[row,col]=(int(m[0]),int(m[1]),int(m[2]))
    cv.imshow("convolution-demo",dst)

    # 函数实现卷积，如果（5，5）改为（1，25），则为垂直方向卷积，（25，1）为水平方向卷积
    blured=cv.blur(image,(25,1),anchor=(-1,-1))
    cv.imshow("blur-demo",blured)

    cv.waitKey(0)
    cv.destroyAllWindows()


def gaussian_blur_demo():
    image=cv.imread("C://Users//86198//Desktop//DataWhale//OpenCV//opencv//data//lena.jpg")
    cv.imshow("input",image)
    g1=cv.GaussianBlur(image,(0,0),15)
    g2=cv.GaussianBlur(image,(15,15),15)
    cv.imshow("GaussianBlur-demo1",g1)
    cv.imshow("GaussianBlur-demo2",g2)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    gaussian_blur_demo()
