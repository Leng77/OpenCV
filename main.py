import cv2 as cv
import numpy as np
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
    image=cv.imread("C://Users//86198\Desktop//DataWhale//OpenCV//opencv//data//butterfly.jpg")
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




if __name__ == '__main__':
   random_demo()
