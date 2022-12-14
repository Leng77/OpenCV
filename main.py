import cv2 as cv
import numpy as np
def color_space_demo():
    image=cv.imread("lena.jpg")
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
    image=cv.imread("lena.jpg")
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
    image1=cv.imread("lena.jpg")
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


if __name__ == '__main__':
   arithmetic_demo()