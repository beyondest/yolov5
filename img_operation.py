import cv2
import os
import moviepy
from moviepy.editor import VideoFileClip
import numpy as np
import os_operation as oso
from threading import Thread
from time import sleep,ctime
import random
import matplotlib.pyplot as plt
import math
import sys

#yuv_range:
#red:152-255,v
#blue:145-255,u



def target_find(img_ori:np.ndarray,armor_color:str,filter_params:list)->tuple:
    '''
    Function:\n
    NO1:find light_bar\n
    NO2:make big rec to cover two light_bar\n
    NO3:use big rec to make a mask to select roi\n
    notice that there may be a lot of roi,so return list\n
    NO4:calculate time and show FPS\n
    return draw_img,roi_single_list,all_time
    
    '''
    img_single,t1=pre_process(img_ori,armor_color)
    big_rec_list,t2=find_big_rec_plus(img_single,filter_params)
    draw_img=draw_big_rec(big_rec_list,img_ori)
    roi_list,t3=pick_up_roi(big_rec_list,img_ori)
    roi_single_list,t4=pre_process2(roi_list,armor_color)
    all_time=t1+t2+t3+t4
    return draw_img,roi_single_list,all_time
###############################################################
def pre_process(img_ori:np.ndarray,armor_color:str)->list:
    '''
    armor_color = 'red' or 'blue'\n
    return img_single,time
    '''
    img_size_yx=(img_ori.shape[0],img_ori.shape[1])
    
    time1=cv2.getTickCount()
    
    dst=cv2.GaussianBlur(img_ori,(3,3),1)
    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2YUV)
    #out of memory error
    y,u,v=cv2.split(dst)
    #blue 145-255
    #red 152-255
    if armor_color=='blue':
        dst=cv2.inRange(u.reshape(img_size_yx[0],img_size_yx[1],1),145,255)
    elif armor_color=='red':
        dst=cv2.inRange(v.reshape(img_size_yx[0],img_size_yx[1],1),152,255)
    else:
        print('armor_color is wrong')
        sys.exit()
    #dst=cv2.medianBlur(dst,13)
    if img_size_yx==(1024,1280):
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    elif img_size_yx==(256,320):
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    else:
        print('imgsize not match, you havent preset preprocess params yet')  
        sys.exit() 
    dst=cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel)
    
    time2=cv2.getTickCount()
    time=(time2-time1)/cv2.getTickFrequency()
    
    
    
    
    
    return dst,time


def find_big_rec_plus(img_single:np.ndarray,
                      filter_params:tuple
                      )->tuple:
    '''
    add trackbar_set\n
    filter_params:
    @params[0]: area_range
    @params[1]: normal_ratio
    @params[2]: strange_ratio1
    @params[3]: strange_ratio2
    @params[4]: shape_like_range
    @params[5]: center_dis_range
    '''
    img_size_yx=(img_single.shape[0],img_single.shape[1])
    out_list=[]
    final_list=[]
    
    
    
    time1=cv2.getTickCount()
    conts,arrs=cv2.findContours(img_single,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
            
    conts=filter_area(conts,filter_params[0])
    #print(len(conts))
    conts=filter_normal(conts,ratio=filter_params[1])
    #print(len(conts))
    conts=filter_strange(conts,ratio=filter_params[2])
    #print(len(conts))

    conts_tuple_list=filter_no_shapelike(conts,filter_params[4],filter_params[4])
    #print(len(conts_tuple_list))
    for i in conts_tuple_list:
        
        if is_center_dis_good(i[0],i[1],filter_params[5]):
        
            big_rec_info=make_big_rec(i[0],i[1])
            #wid and hei cant < 1
            if big_rec_info[2]>1 and big_rec_info[3]>1:
                
                out_list.append(big_rec_info[4])
        
        
        
    out_list=filter_strange(out_list,filter_params[3])
    out_list=expand_rec_wid(out_list,expand_rate=2,img_size_yx=img_size_yx) 
    
   
        
        
    time2=cv2.getTickCount()
    time=(time2-time1)/cv2.getTickFrequency()
    return out_list ,time 

def find_big_rec(img_single:np.ndarray)->list:
    '''
    big_rec is expanded already\n
    return big_rec_list,time
    '''
    
    img_size_yx=(img_single.shape[0],img_single.shape[1])
    out_list=[]
    if img_size_yx==(1024,1280):
        mode='big'
    elif img_size_yx==(256,320):
        mode='small'
    else:
        print('img_size not match, you havent preset params of find_big_rec yet')
        sys.exit()
    
    
    
    time1=cv2.getTickCount()
    conts,arrs=cv2.findContours(img_single,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if mode=='big':
        
        if len(conts)!=-1:
            
            conts,arrs=filter_area(conts,arrs,(500,2500))
            #print(len(conts))
            conts,arrs=filter_normal(conts,arrs,ratio=2)
            #print(len(conts))
            conts,arrs=filter_strange(conts,arrs,ratio=5)
            #print(len(conts))
        
            conts_tuple_list=filter_no_shapelike(conts,(0.4,1.6),(0.4,1.6))
            #print(len(conts_tuple_list))
            for i in conts_tuple_list:
                
                if iscenternear(i[0],i[1],600):
                
                    big_rec_info=make_big_rec(i[0],i[1])
                    #wid and hei cant < 1
                    if big_rec_info[2]>1 and big_rec_info[3]>1:
                        
                        out_list.append(big_rec_info[4])
        
        else:
            big_rec_info=make_big_rec(conts[0],conts[1])
            out_list.append(big_rec_info[4])
        
    
        out_list=expand_rec_wid(out_list,expand_rate=2,img_size_yx=img_size_yx) 
    
    elif mode=='small':
        #small_params = 1/4 * big_params
        if len(conts)!=-1:
            
            conts,arrs=filter_area(conts,arrs,(1,400))
            #print(len(conts))
            conts,arrs=filter_normal(conts,arrs,ratio=1.2)
            #print(len(conts))
            conts,arrs=filter_strange(conts,arrs,ratio=10)
            #print(len(conts))
        
            conts_tuple_list=filter_no_shapelike(conts,(0.5,1.5),(0.5,1.5))
            #print(len(conts_tuple_list))
            for i in conts_tuple_list:
                
                if iscenternear(i[0],i[1],80):
                    
                    big_rec_info=make_big_rec(i[0],i[1])
                    
                    #wid and hei cant <1
                    if big_rec_info[2]>1 and big_rec_info[3]>1:
                    
                        out_list.append(big_rec_info[4])
        
        else:
            big_rec_info=make_big_rec(conts[0],conts[1])
            out_list.append(big_rec_info[4])
        
    
        out_list=expand_rec_wid(out_list,expand_rate=2,img_size_yx=img_size_yx)  
        
        
        
    time2=cv2.getTickCount()
    time=(time2-time1)/cv2.getTickFrequency()
    return out_list ,time 

def draw_big_rec(big_rec_list,img_bgr:np.ndarray)->np.ndarray:
    '''
    return img_copy_bgr
    '''
    if img_bgr.shape[2]==1:
        img_bgr=cv2.cvtColor(img_bgr,cv2.COLOR_GRAY2BGR)
    img_copy=img_bgr.copy()
    for i in big_rec_list:
        draw_cont(img_copy,i,2,3)
    return img_copy
    
def pick_up_roi(big_rec_list,img_ori:np.ndarray)->list:
    '''
    return roi_list,time
    '''
    img_size_yx=(img_ori.shape[0],img_ori.shape[1])
    t1=cv2.getTickCount()
    roi_list=[]
    
    background=np.zeros((img_size_yx[0],img_size_yx[1]),dtype=np.uint8)
    for i in big_rec_list:
        i=i.reshape(-1,1,2)
        back_copy=background.copy()
        img_copy=img_ori.copy()
        dst=img_copy
        mask=cv2.fillPoly(back_copy,[i],255)
        dst=cv2.bitwise_and(img_copy,img_copy,mask=mask)
        roi_list.append(dst)
    if len(big_rec_list)==0:
        bgr=cv2.cvtColor(background,cv2.COLOR_GRAY2BGR)
        roi_list.append(bgr)
    t2=cv2.getTickCount()
    time= (t2-t1)/cv2.getTickFrequency()
    return roi_list,time


def pre_process2(roi_list,armor_color:str)->list:
    '''
    preprocess for finding number\n
    return roi_single_list,time
    '''
    out=[]
    
    t1=cv2.getTickCount()
    for i in roi_list:
        img_size_yx=(i.shape[0],i.shape[1])
        dst=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        #y,u,v=cv2.split(dst)
        #127 for red
        #90 for blue 
        #dst=cv2.inRange(y.reshape(img_size_yx[0],img_size_yx[1],1),77,158)
        if armor_color=='red':
            ret,dst=cv2.threshold(dst,127,255,cv2.THRESH_BINARY)
        elif armor_color=='blue':    
            ret,dst=cv2.threshold(dst,90,255,cv2.THRESH_BINARY)
        out.append(dst)
    t2=cv2.getTickCount()
    time=(t2-t1)/cv2.getTickFrequency()
    return out,time
    
#####################################################################
def plt_show0(img):
    '''
    show 3 channels bgr in plt
    '''
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()
    
def plt_show(img):
    '''
    show single channel img
    '''
    plt.imshow(img,cmap='gray')
    plt.show()

def add_text(img_bgr:np.ndarray,
             name:str,
             value,
             pos:tuple=(-1,-1),
             font:int=cv2.FONT_HERSHEY_COMPLEX,
             color:tuple=(100,200,200),
             )->np.ndarray:
    '''
    show name:value on the position of pos(x,y)\n
    if pos =(-1,-1), then auto position\n
    return img_bgr
    '''
    img_size_yx=(img_bgr.shape[0],img_bgr.shape[1])
    if pos==(-1,-1):
        
        pos=(round(img_size_yx[1]/10),round(img_size_yx[0]/10))
    if img_size_yx==(1024,1280):
        scale_size=2
    elif img_size_yx==(256,320):
        scale_size=0.5
    else:
        print('img_size not match, you havent preset for add_text yet')  
    thickness=round(3/(1024*1280)*(img_size_yx[0]*img_size_yx[1]))
    dst=cv2.putText(img_bgr,f'{name}:{value}',pos,font,scale_size,color,thickness)
    return dst
    
###################################################################
def gray_guss(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def get_edgebin2(ori_img:np.ndarray)->np.ndarray:
    '''by canny,50-100'''
    img_copy=ori_img.copy()
    img_gray=gray_guss(img_copy)
    img_blur=cv2.GaussianBlur(img_gray,(5,5),1)
    img_canny=cv2.Canny(img_blur,50,100)
    img_canny=cv2.GaussianBlur(img_canny,(3,3),1)
    ret,img_edge_bin=cv2.threshold(img_canny,0,255,cv2.THRESH_OTSU)
    img_edge_bin=cv2.medianBlur(img_edge_bin,3)
    return img_edge_bin

def get_edgebin1(ori_img:np.ndarray)->np.ndarray:
    '''by sobel'''
    img_copy=ori_img.copy()
    img_gray=gray_guss(img_copy)
    sobel_x=cv2.Sobel(img_gray,cv2.CV_16S,1,0,ksize=3)
    sobel_y=cv2.Sobel(img_gray,cv2.CV_16S,0,1,ksize=3)
    sobel_x=cv2.convertScaleAbs(sobel_x)
    sobel_y=cv2.convertScaleAbs(sobel_y)
    img_sobel=cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
    img_sobel=cv2.GaussianBlur(img_sobel,(3,3),1)
    ret,img_thresh=cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU)
    img_thresh=cv2.medianBlur(img_thresh,5)
    ker_rec=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    img_close=cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,ker_rec,iterations=1)

    img_edge_bin=cv2.medianBlur(img_close,5)
    return img_edge_bin
################################################################3
def make_edge(abs_path:str,out_path:str,fmt:str='png'):
    ori_img=cv2.imread(abs_path)
    edge_bin=get_edgebin1(ori_img)
    ori_name=oso.get_name(abs_path)
    cv2.imwrite(os.path.join(out_path,ori_name+'edgebin'+'.'+fmt),edge_bin)

def show_duration(abs_path):
    vd=VideoFileClip(abs_path)
    print(f'{abs_path} duration={vd.duration}')
    
def cut_video(abs_path:str,
              out_path:str,
              t_start:tuple,
              t_end=None,
              interval_sec:int=10,
              name:str='sub',
              fmt:str='avi'
              ):
    vd=VideoFileClip(abs_path)  
    ori_name=oso.get_name(abs_path)
    subclip = vd.subclip(30, -30)
    subclip.write_videofile(os.path.join(out_path,ori_name+name+''+'.'+fmt))





def draw_cont(ori_copy:np.ndarray,cont:np.ndarray,color:int=2,thick:int=2):
    '''
    @cont:  np.array(left_down,left_up,right_up,right_down)
    @color=0,1,2=blue,green,red
    @thick:   defalt is 2
    '''
    colorlist=[(255,0,0),(0,255,0),(0,0,255)]
    cv2.drawContours(ori_copy,[cont],-1,colorlist[color],thick)
    return 0

def make_cont(img,thr_area:int=500):

    edge_bin=get_edgebin1(img)
    
    conts,arrs=cv2.findContours(edge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    img_copy=img.copy()
    for i in conts:
        area=cv2.contourArea(i)
        if area>thr_area:
            rec_points=getrec_info(i)[4]
            draw_cont(img_copy,rec_points,2)
    return img_copy


def getframe_info(vd_abspath:str)->tuple:
    '''fps,frame_count,frame_width,frame_height=getframe_info()'''
    vd=cv2.VideoCapture(vd_abspath)
    Fps=vd.get(cv2.CAP_PROP_FPS)
    frame_count=vd.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height=vd.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width=vd.get(cv2.CAP_PROP_FRAME_WIDTH)
    vd.release()
    return Fps,frame_count,frame_width,frame_height

def readframe(abs_path:str,
              out_path:str,
              fmt:str,
              interval:int=5,
              name:str='frame',
              start:int=0,
              length:int=0):
    '''cv2 get started from 0 frame'''
    vd_name=oso.get_name(abs_path)
    vd=cv2.VideoCapture(abs_path)
    vd.set(cv2.CAP_PROP_POS_FRAMES,start)
    if length==0:
        length=vd.get(cv2.CAP_PROP_FRAME_COUNT)
    count=start-1
    #time check
    print(f' {start} start as {ctime()}')
    while True:
        count+=1
        iscaptured=vd.grab()
        if not iscaptured or count==start+length:
            break
        if count%interval==0:
            iscaptured,frame=vd.retrieve()
            if frame is not None:
                cv2.imwrite(os.path.join(out_path,vd_name+name+'{}.'.format(int(count))+fmt),frame)  
            else:
                break
    #time check
    print(f' {start} end as {ctime()}')
    vd.release()

def readframe_pro(vd_abspath:str,
                  out_path:str,
                  fmt:str='png',
                  interval:int=10,
                  name:str='frame',
                  threads:int=6,
                  start_frame:int=100,
                  end_frame:int=4000)->None:
    '''read frame faster by multi-thread,
        frame range is (0,frame_count-1)'''
    vd=cv2.VideoCapture(vd_abspath)
    frame_count=vd.get(cv2.CAP_PROP_FRAME_COUNT)
    if end_frame == 0 or end_frame>=frame_count:
        end_frame=frame_count
    #length means how long each thread deal with
    length=(end_frame-start_frame)//threads+1
    start_list=[start_frame+i*length for i in range(threads)]
    #multi-thread, after test, 6 runs fastest,10 seconds for 200 frames,1 runs lowest, 35 seconds for 200 frames
    t=[]
    for i in range(threads):
        t.append(Thread(target=readframe,args=(vd_abspath,out_path,fmt,interval,name,start_list[i],length)))
    for i in range(threads):
        t[i].start()
    

    
    return None


def find_inorder(points_list:list)->list:
    for i in range(3):
        if points_list[i][0]>points_list[i+1][0]:
            points_list[i],points_list[i+1]=points_list[i+1],points_list[i]
    if points_list[0][1]<points_list[1][1]:
        points_list[0],points_list[1]=points_list[1],points_list[0]
    if points_list[2][1]<points_list[3][1]:
        points_list[2],points_list[3]=points_list[3],points_list[2]
    points_list[1],points_list[2]=points_list[2],points_list[1]
    return points_list
            
     
#1000,max3,max2 is best up to now
def make_findcont_transform(abs_path:str,out_path:str,fmt:str='png',thr_area:int=1000):
    p_list = []  
    dst_point = (1000, 1000)  
    img=cv2.imread(abs_path)
    img_size_yx=(img.shape[0],img.shape[1])
    img2=img.copy()
    ori_name=oso.get_name(abs_path)
    edge_bin=get_edgebin1(img)
    rec_points=[]
    conts,arrs=cv2.findContours(edge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for i in conts:
        area=cv2.contourArea(i)
        if area>thr_area:
            rec_points.append(getrec_info(i)[4])
    if len(rec_points)==0:
        wid=img_size_yx[1]
        hei=img_size_yx[0]
        x1=random.randint(0,wid//3)
        y1=random.randint(0,hei//3)
        x2=random.randint(wid-wid//3,wid-1)
        y2=random.randint(0,hei//3)
        x3=random.randint(0,wid//3)
        y3=random.randint(hei-hei//3,hei-1)
        x4=random.randint(wid-wid//3,wid-1)
        y4=random.randint(hei-hei//3,hei-1)
        p_list =[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    elif len(rec_points)==1:
        
        p_list= [rec_points[0][0],rec_points[0][1],rec_points[0][3],rec_points[0][2]]
        p_list=find_inorder(p_list)
    elif len(rec_points)>1:
        area_list=[]
        for i in rec_points:
            area_list.append(cv2.contourArea(i))
        compare_list=sorted(area_list,reverse=True)
        max1=area_list.index(compare_list[0])
        max2=area_list.index(compare_list[1])
        max1=rec_points[max1]
        max2=rec_points[max2]
        
        if len(rec_points)>2:
            max3=area_list.index(compare_list[2])
            max3=rec_points[max3]
            fine=max3
            p_list= [fine[0],fine[1],fine[3],fine[2]]
            p_list=find_inorder(p_list)
        else:
            fine=max2
            p_list= [fine[0],fine[1],fine[3],fine[2]]
            p_list=find_inorder(p_list)
    pts1 = np.float32(p_list)
    pts2 = np.float32([[0, 0], [dst_point[0], 0], [0, dst_point[1]], [dst_point[0], dst_point[1]]])
    dst = cv2.warpPerspective(img2, cv2.getPerspectiveTransform(pts1, pts2), dst_point)
    dst=cv2.flip(dst,0)
    cv2.imwrite(os.path.join(out_path,ori_name+'trans'+'.'+fmt),dst)

def make_bin(abs_path:str,out_path:str,fmt:str='png',name:str='bin'):
    img = cv2.imread(abs_path)
    ori_name=oso.get_name(abs_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, img_bin= cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(out_path,ori_name+name+'.'+fmt),img_bin)

def make_noise(abs_path:str,
               out_path:str,
               fmt:str='png'):
    '''make random white circle as noise, the radius of it is 
    1/7 of height, 1/3 possibility run'''
    judge=random.randint(1,3)
    if judge==1:
        img=cv2.imread(abs_path)
        ori_name=oso.get_name(abs_path)
        out=os.path.join(out_path,ori_name+'.'+fmt)
        height=img.shape[0]
        wid=img.shape[1]
        radius=height//7
        x=random.randint(0,wid-1)
        y=random.randint(0,height-1)
        cv2.circle(img,(x,y),radius,(255,255,255),-1)  
        cv2.imwrite(out,img)
        
def make_rotate(abs_path:str,
                out_path:str,
                angel:int=180,
                fmt:str='png'
                ):
    '''angel=90 clockwise/-90 counterclockwise/180'''
    if angel==90:
        cvcode=cv2.ROTATE_90_CLOCKWISE
    elif angel==-90:
        cvcode=cv2.ROTATE_90_COUNTERCLOCKWISE
    elif angel==180:
        cvcode=cv2.ROTATE_180
    img=cv2.imread(abs_path)
    img_rotate=cv2.rotate(img,cvcode)
    ori_name=oso.get_name(abs_path)
    out=os.path.join(out_path,ori_name+'.'+fmt)
    cv2.imwrite(out,img_rotate)
    
def rename_distinguish(abs_path:str,out_path:str,fmt:str='png'):
    '''shape[1]>shape[0] is big, if use muitiwork, it out_path 
    must be configed , cause even out_name='',it has '\\' in last 
    location'''
    img=cv2.imread(abs_path)
    if img.shape[1]>img.shape[0]:
        out_name='d'
        
    else:
        out_name='x'
        
    dir_path=os.path.split(out_path)[0]
    out=dir_path+out_name   
    ori_name=oso.get_name(abs_path)
    out=os.path.join(out,ori_name+out_name+'.'+fmt)
    os.rename(abs_path,out)
    
def make_cut(abs_path:str,out_path:str,fmt:str='png',suffix_name:str='cut'):
    img=cv2.imread(abs_path)
    ori_name=oso.get_name(abs_path)
    wid=img.shape[1]
    k=145/693
    cut_wid=round(wid*k)
    img2=img.copy()
    dst=img2[:,cut_wid:wid-cut_wid,:]
    out=os.path.join(out_path,ori_name+suffix_name+'.'+fmt)
    cv2.imwrite(out,dst)

##################################################################

def getrec_info(cont:np.ndarray)->tuple:
    '''
    input ori_cont,\n
    return:\n
    center_x,\n
    center_y,\n
    width,\n
    height,\n
    rec_points(its shape is same as one cont),\n
    rec_area
    '''
    ret=cv2.minAreaRect(cont)
    bo=cv2.boxPoints(ret)
    bo=np.int0(bo)
    rec_area=cv2.contourArea(bo)
    rec_area=abs(rec_area)
    return (ret[0][0],ret[0][1],ret[1][0],ret[1][1],bo,rec_area)
    
def getimg_info(abs_path:str)->tuple:
    '''return img_size,shape,dtype'''
    img=cv2.imread(abs_path)
    img_size=img.size
    img_shape=img.shape
    img_dtype=img.dtype
    return img_size,img_shape,img_dtype

def pertrans(img:np.ndarray)->np.ndarray:
    '''watch the size'''
    img_size_yx=(img.shape[0],img.shape[1])
    pts1=([0,0],[img_size_yx[1],0],[0,img_size_yx[0]],[img_size_yx[1],img_size_yx[0]])
    pts1=np.float32(pts1)
    dst_point=[img_size_yx[1],img_size_yx[0]]
    pts2 = np.float32([[0, 0], [dst_point[0], 0], [0, dst_point[1]], [dst_point[0], dst_point[1]]])
    dst = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts1, pts2), dst_point)
    return dst

def image_binary(image):
    max_value=float(image.max())
    min_value=float(image.min()) 

    ret=max_value-(max_value-min_value)/2
    ret,thresh=cv2.threshold(image,ret,255,cv2.THRESH_BINARY)
    return thresh
    

def gray_stretch(image):
    max_value=float(image.max())
    min_value=float(image.min())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j]=(255/(max_value-min_value)*image[i,j]-(255*min_value)/(max_value-min_value))
    return image
################################################################################3
def filter_area(conts_list:list,area_range:tuple=(100,500))->list:
    '''
    filter rec_area that is not in range,\n
    return conts_list, arrs_list
    '''
    conts_out=[]
    
    for i in range(len(conts_list)):
        rec_cont=cv2.minAreaRect(conts_list[i])
        rec_cont=cv2.boxPoints(rec_cont)
        rec_cont=np.int0(rec_cont)
        rec_area=cv2.contourArea(rec_cont)
        rec_area=abs(rec_area)
        
        if area_range[0]<rec_area<area_range[1]:
            
            conts_out.append(conts_list[i])
        
    return conts_out

def filter_nohavechild(conts_list:list,arrs_list:list)->tuple:
    '''filter conts which does not have a child_cont,leave the outest cont,notice that these outest do have child'''
    conts_out=[]
    arrs_out=[[]]
    for i in range(len(arrs_list[0])):
        #if have child cont
        if arrs_list[0][i][2]!=-1:
            conts_out.append(conts_list[i])
            arrs_out[0].append(arrs_list[0][i])
    return conts_out,arrs_out

def filter_havechild(conts_list:list,arrs_list:list)->tuple:
    '''filter conts which has a child_cont, leave the innerest cont,notice that these innerests may not have parent'''
    conts_out=[]
    arrs_out=[[]]
    for i in range(len(arrs_list[0])):
        #if do not have child cont
        if arrs_list[0][i][2]==-1:
            conts_out.append(conts_list[i])
            arrs_out[0].append(arrs_list[0][i])
    return conts_out,arrs_out

def find_original_num(ori_list:list,conts:np.ndarray)->int:
    '''return the nums in ori_conts_list'''
    for i in range(len(ori_list)):
        if ori_list[i].shape==conts.shape:
            if (ori_list[i]==conts).all():
                return i
#false
def find_current_num(cur_list:list,conts:np.ndarray)->int:
    '''return the nums in cur_conts_list'''
    for i in range(len(cur_list)):
        if i.shape==conts.shape:
            if (i==conts).all():
                return cur_list.index(i) 
       
   
    
def filter_strange(conts_list:list,ratio:float=5)->tuple:
    '''
    filter the shape looks so strange that their rec is too thin
    '''
    conts_out=[]

    for i in range(len(conts_list)):
        info_tuple=getrec_info(conts_list[i])
        x=info_tuple[0]
        y=info_tuple[1]
        wid=info_tuple[2]
        hei=info_tuple[3]
        proportion=wid/hei
        if proportion>1:
            if proportion>ratio:
                continue
        else:
            if 1/proportion>ratio:
                continue
        
        conts_out.append(conts_list[i])
    return conts_out

def filter_normal(conts_list,ratio:float=1.5)->tuple:
    '''
    filter the shape looks too normal that their rec is nearly square 
    '''
    conts_out=[]

    for i in range(len(conts_list)):
        x,y,wid,hei,_,_=getrec_info(conts_list[i])
        proportion=wid/hei
        if proportion>1:
            if proportion<ratio:
                continue
        else:
            if 1/proportion<ratio:
                continue
        
        conts_out.append(conts_list[i])

    return conts_out

def filter_no_shapelike(conts_list,area_near_range:tuple=(0.5,1.5),ratio_near_range:tuple=(0.5,1.5))->list:
    '''
    filter conts that has no shapelike cont ,\n
    ratio_range describes what is near;\n
    return list of tuple, each tuple is a pair of cont!!!
    '''
    
    conts_out=[]
    tuple_list=[]
    
    for i in range(len(conts_list)):
        center_x,center_y,wid,hei,rec_points,rec_area=getrec_info(conts_list[i])
        for j in tuple_list:
            #first rec_area is near
            if area_near_range[0]<rec_area/j[0]<area_near_range[1]:
                #second wid/hei is near:
                if ratio_near_range[0]<wid/hei/j[1]<ratio_near_range[1]:
                    conts_out.append((j[2],conts_list[i]))
        tuple_list.append((rec_area,wid/hei,conts_list[i]))
    return conts_out



def set_grades(conts_list:list)->dict:
    grades_dict={i:0 for i in range(len(conts_list))}
    return grades_dict

    
def color_test(conts_list:list,
               grades_dict:dict,
               ori_bin:np.ndarray,
               white_ratio_range:list=[0.6,1],
               center_wid:int=4):
    '''@brief access by white_ratio_range and center white judge
    white_ratio=white/whole'''
    for each in range(len(conts_list)):
        #create minrec,then calculate the color
        center_x,center_y,wid,hei,rec_points,_=getrec_info(conts_list[each])
        leftup_x=rec_points[0][0]
        leftup_y=rec_points[0][1]
        leftdown_x=rec_points[3][0]
        leftdown_y=rec_points[3][1]
        if leftdown_x==leftup_x:
            slope=0
        else:
            slope=(leftdown_y-leftup_y)/(leftdown_x-leftup_x)
        white_sum=0
        black_sum=0
        for i in range(int(hei)):
            if leftup_y+i>1023:
                break
            for j in range(int(wid)):
                if round(leftup_x+slope*i+j)>1279 or round(leftup_x+slope*i+j)<0:
                    break
                if ori_bin[leftup_y+i][round(leftup_x+slope*i)+j]==255:
                    white_sum+=1
                else:
                    black_sum+=1      
        #create center square,then calculate the color
        sum_center=0
        for i in range(center_wid):
            if center_y-center_wid/2+i<0 or center_y-center_wid/2+i>1023:
                break
            for j in range(center_wid):
                if center_x-center_wid/2+j<0 or center_x-center_wid/2+j>1279:
                    break
                sum_center+=ori_bin[round(center_y-center_wid/2+i)][round(center_x-center_wid/2+j)]
        area_center=center_wid*center_wid
        
        #begin access grade
        #white_ratio +10
        white_ratio=white_sum/(white_sum+black_sum)
        if white_ratio>white_ratio_range[0] and white_ratio<white_ratio_range[1]:
            grades_dict[each]+=10
        #white_center +5
        if sum_center>area_center*255*0.8:
            grades_dict[each]+=5

def parent_test(conts_list:list,
                arrs_list:list,
                grade_dict:dict,
                ori_cont_list:list,
                area_ratio:float=1.2):
    '''area_ratio=parent_area/child_area'''
    for i in range(len(arrs_list[0])):
        #if have parent:non solid always have parent
        if arrs_list[0][i][3]!=-1:
            parent_ori_num=arrs_list[0][i][3]
            parent_cont=ori_cont_list[parent_ori_num]
            child_cont=conts_list[i]
            
            parent_area=cv2.contourArea(parent_cont)
            child_area=cv2.contourArea(child_cont)
            #inner-outer match +10
            if parent_area<child_area*area_ratio and parent_area>child_area:
                grade_dict[i]+=10

def child_test(conts_list:list,
                arrs_list:list,
                grade_dict:dict,
                ori_cont_list:list,
                area_ratio:float=1.2):
    '''area_ratio=parent_area/child_area'''
    for i in range(len(arrs_list[0])):
        #if have child: enclosed shape usually have child
        if arrs_list[0][i][2]!=-1:
            child_ori_num=arrs_list[0][i][2]
            child_cont=ori_cont_list[child_ori_num]
            parent_cont=conts_list[i]
            
            parent_area=cv2.contourArea(parent_cont)
            child_area=cv2.contourArea(child_cont)
            #inner-outer match +10
            if parent_area<child_area*area_ratio and parent_area>child_area:
                grade_dict[i]+=10

def goodshape_test(conts_list:list,
                   grade_dict:dict,
                   area_ratio:float=0.7):
    '''area_ratio=cont_area/minrec_area'''
    for i in range(len(conts_list)):
        cont_area=cv2.contourArea(conts_list[i])
        rec_area=getrec_info(conts_list[i])[5]
        #good shape +10
        if cont_area/rec_area>area_ratio:
            grade_dict[i]+=10

def search_inner(ori_conts_list:list,
                 ori_arrs_list:list,
                 ori_bin:np.ndarray,
                 out_nums:int=3)->list:
    '''search for the best inner conts,return out_nums conts in list ,
    you may need to change some params to adapt to different inner shapes,
    actually return innerest conts'''
    if len(ori_conts_list)<=3:
        return ori_conts_list
    bigger_conts,bigger_arrs=filter_area(ori_conts_list,ori_arrs_list,200)
    if len(bigger_conts)<=3:
        return bigger_conts
    inner_conts,inner_arrs=filter_havechild(bigger_conts,bigger_arrs)
    if len(inner_conts)<=3:
        return inner_conts
    normal_conts,normal_arrs=filter_strange(inner_conts,inner_arrs,5)
    if len(normal_conts)<=3:
        return normal_conts
    grades_dict=set_grades(normal_conts)
    #you cannot filter normal_conts cause grade_dict is set on nums of range(len(normal_conts))
    color_test(normal_conts,grades_dict,ori_bin)
    parent_test(normal_conts,normal_arrs,grades_dict,ori_conts_list)
    goodshape_test(normal_conts,grades_dict)
    out_list=[]
    key_list=list(grades_dict.keys())
    value_list=list(grades_dict.values())
    for i in range(out_nums):
        max_index=value_list.index(max(value_list))
        out_list.append(normal_conts[key_list[max_index]])
        key_list.remove(key_list[max_index])
        value_list.remove(value_list[max_index])
    return out_list
        
def search_outer(ori_conts_list:list,
                 ori_arrs_list:list,
                 ori_bin:np.ndarray,
                 out_nums:int=3)->list:
    '''search for the best outer conts, return out_nums conts in list,
    actually return outest conts'''
    if len(ori_conts_list)<=3:
        return ori_conts_list
    bigger_conts,bigger_arrs=filter_area(ori_conts_list,ori_arrs_list,1500)
    if len(bigger_conts)<=3:
        return bigger_conts
    outer_conts,outer_arrs=filter_nohavechild(bigger_conts,bigger_arrs)
    if len(outer_conts)<=3:
        return outer_conts
    normal_conts,normal_arrs=filter_strange(outer_conts,outer_arrs,4)
    if len(normal_conts)<=3:
        return normal_conts
    grades_dict=set_grades(normal_conts)
    color_test(normal_conts,grades_dict,ori_bin,[0.05,0.4])
    child_test(normal_conts,normal_arrs,grades_dict,ori_conts_list,1.1)
    goodshape_test(normal_conts,grades_dict,0.9)
    out_list=[]
    key_list=list(grades_dict.keys())
    value_list=list(grades_dict.values())
    for i in range(out_nums):
        max_index=value_list.index(max(value_list))
        out_list.append(normal_conts[key_list[max_index]])
        key_list.remove(key_list[max_index])
        value_list.remove(value_list[max_index])
    return out_list

def isrelative(cont1:np.ndarray,
               cont2:np.ndarray,
               ori_conts_list:list,
               ori_arrs_list:list)->int:
    '''return 0 for none, 1 for cont1 is child, 2 for cont1 is parent'''
    ori_num1=find_original_num(ori_conts_list,cont1)
    ori_num2=find_original_num(ori_conts_list,cont2)
    if ori_num1==ori_arrs_list[0][ori_num2][2] or ori_num2==ori_arrs_list[0][ori_num1][3]:
        return 1
    elif ori_num1==ori_arrs_list[0][ori_num2][3] or ori_num2==ori_arrs_list[0][ori_num1][2]:
        return 2
    else:
        return 0

def iscenternear(cont1:np.ndarray,
                 cont2:np.ndarray,
                 distance:int=50)->bool:
    '''
    <distance return true\n
    else return false
    '''
    x1,y1,_,_,_,_=getrec_info(cont1)
    x2,y2,_,_,_,_=getrec_info(cont2)
    dis=((x1-x2)**2+(y1-y2)**2)**0.5
    if dis<distance:
        return True
    else:
        return False

def is_center_dis_good(cont1:np.ndarray,
                       cont2:np.ndarray,
                       center_dis_range:tuple):
    '''
    return True if center_dis is in range
    '''
    x1,y1,_,_,_,_=getrec_info(cont1)
    x2,y2,_,_,_,_=getrec_info(cont2)
    dis=((x1-x2)**2+(y1-y2)**2)**0.5
    if center_dis_range[0]<dis<center_dis_range[1]:
        return True
    else:
        return False
    
    

def walk_until_white(begin_x:int,
                     begin_y:int,
                     slope:float,
                     img_edge_bin:np.ndarray,
                     direction:int=0)->tuple:
    '''direction=0 for x plus 1 direction, =1 for x minus dirction
    ,   return x,y'''
    x=begin_x
    y=begin_y
    if direction==0:
        while True:
            x+=1
            y+=slope
            if x>1279 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return min(x,1279),y
                elif y<0:
                    y=0
                    return min(x,1279),y
                elif x>1279:
                    x=1279
                    return x,y
            if img_edge_bin[round(y)][x]==255:
                return x,round(y)
    if direction==1:
        while True:
            x-=1
            y+=slope
            if x<0 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return max(x,0),y
                elif y<0:
                    y=0
                    return max(x,0),y
                elif x<0:
                    x=0
                    return x,y
            if img_edge_bin[round(y)][x]==255:
                return x,round(y) 

 
def walk_until_black(begin_x:int,begin_y:int,slope:float,img_edge_bin:np.ndarray,
                     direction:int=0)->tuple:
    '''direction =0 for x+=1, or 1 for x-=1'''      
    x=begin_x
    y=begin_y
    if direction==0:
        while True:
            x+=1
            y+=slope
            if x>1279 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return min(x,1279),y
                elif y<0:
                    y=0
                    return min(x,1279),y
                elif x>1279:
                    x=1279
                    return x,y
            if img_edge_bin[round(y)][x]==0:
                return x,round(y)
    if direction==1:
        while True:
            x-=1
            y+=slope
            if x<0 or y>1023 or y<0:
                if y>1023:
                    y=1023
                    return max(x,0),y
                elif y<0:
                    y=0
                    return max(x,0),y
                elif x<0:
                    x=0
                    return x,y
            if img_edge_bin[round(y)][x]==0:
                return x,round(y)

def walk_until_dis(begin_x:int,
                   begin_y:int,
                   slope:float,
                   distance:float,
                   direction:str='right',
                   img_size_yx:tuple=(1024,1280)
                   
                    )->list:
    '''direction =right for x+=1, or left for x-=1,
    return x,y'''      
    x=begin_x
    y=begin_y
    x_range=(0,img_size_yx[1])
    y_range=(0,img_size_yx[0])
    if math.isinf(slope) or math.isnan(slope):
        print('slope out')
        return [x,y]
    theta=math.atan(abs(slope))
    delta_x=distance*math.cos(theta)
    
    delta_y=distance*math.sin(theta)
    if direction=='right':
        
        x+=delta_x
        y=y+delta_y if slope>0 else y-delta_y
        #check range 
        if x>x_range[1]:
            x=x_range[1]
        if y>y_range[1]:
            y=y_range[1]
        if y<y_range[0]:
            y=y_range[0]
        return [x,round(y)]
    if direction=='left':
        x-=delta_x
        y=y+delta_y if slope<0 else y-delta_y
        #check range
        if x<x_range[0]:
            x=x_range[0]
        if y>y_range[1]:
            y=y_range[1]
        if y<y_range[0]:
            y=y_range[0]
        return [x,round(y)]

def make_plate(cont:np.ndarray,img_edge_bin:np.ndarray)->np.ndarray:
    '''make the plate yourself by innercont, notice that img_edge_bin has to be dealed with medianblur, then binary'''
    
    center_x,center_y,wid,hei,corners,_=getrec_info(cont)
    x1,y1=corners[0]
    x2,y2=corners[1]
    x3,y3=corners[2]
    x4,y4=corners[3]
    slope12=(y2-y1)/(x2-x1)
    slope12_=(-1)/slope12
    x12_=round((x1+x2)/2)
    y12_=round((y1+y2)/2)
    x23_=round((x2+x3)/2)
    y23_=round((y2+y3)/2)
    x34_=round((x3+x4)/2)
    y34_=round((y3+y4)/2)
    x14_=round((x1+x4)/2)
    y14_=round((y1+y4)/2)
    #get the lengths to be lengthen
    x12,y12=walk_until_black(x12_,y12_,slope12_,img_edge_bin,0)
    x12,y12=walk_until_white(x12,y12,slope12_,img_edge_bin,0)
    x23,y23=walk_until_black(x23_,y23_,slope12,img_edge_bin,0)
    x23,y23=walk_until_white(x23,y23,slope12,img_edge_bin,0)
    x34,y34=walk_until_black(x34_,y34_,slope12_,img_edge_bin,1)
    x34,y34=walk_until_white(x34,y34,slope12_,img_edge_bin,1)
    x14,y14=walk_until_black(x14_,y14_,slope12,img_edge_bin,1)
    x14,y14=walk_until_white(x14,y14,slope12,img_edge_bin,1)
    
    points=[[x12,y12],[x23,y23],[x34,y34],[x14,y14]]
    begins=[[x12_,y12_],[x23_,y23_],[x34_,y34_],[x14_,y14_]]
    dis_list=[0,0,0,0]
    for i in range(len(points)):
        if i!=[0,0]:
            dis_list[i]=((points[i][0]-begins[i][0])**2+(points[i][1]-begins[i][1])**2)**0.5
    #begin to access which side to lengthen
    grade12=0
    grade23=0
    #compare uses as a good range
    compare=(wid+hei)/2
    if dis_list[0]!=0 and dis_list[2]!=0:
        #if lengths ro be lengthen are match,grade+10
        if (dis_list[0]-dis_list[2])**2<225:
            grade12+=10
        #if lengths to be lengthen are in a good range,grade+5
        if dis_list[0]<compare and dis_list[2]<compare:
            grade12+=5
        #if both out of range,grade-10
        if dis_list[0]>compare*2.5 and dis_list[2]>compare*2.5:
            grade12-=10
        #if a side out of range,grade-5
        elif dis_list[0]>compare*2.5 or dis_list[2]>compare*2.5:
            grade12-=5
    if dis_list[1]!=0 and dis_list[3]!=0:
        if (dis_list[1]-dis_list[3])**2<225:
            grade23+=10
        if dis_list[1]<compare and dis_list[3]<compare:
            grade23+=5
        if dis_list[1]>compare*2.5 and dis_list[3]>compare*2.5:
            grade23-=10
        elif dis_list[1]>compare*2.5 or dis_list[3]>compare*2.5:
            grade23-=5
    #if 23 side should be lengthen
    if grade12>grade23:
        if grade12>5:
            meandis=(dis_list[0]+dis_list[2])/2
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12_,meandis,0)
            out_points[1]=walk_until_dis(x2,y2,slope12_,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12_,meandis,1)
            out_points[3]=walk_until_dis(x4,y4,slope12_,meandis,1)
            out_points=np.array(out_points)
            return out_points
        else:
            meandis=compare
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12_,meandis,0)
            out_points[1]=walk_until_dis(x2,y2,slope12_,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12_,meandis,1)
            out_points[3]=walk_until_dis(x4,y4,slope12_,meandis,1)
            out_points=np.array(out_points)
            return out_points
    #if 12 side should be lengthen
    elif grade12<grade23:
        if grade23>5:
            meandis=(dis_list[1]+dis_list[3])/2
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12,meandis,1)
            out_points[1]=walk_until_dis(x2,y2,slope12,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12,meandis,0)
            out_points[3]=walk_until_dis(x4,y4,slope12,meandis,1)
            out_points=np.array(out_points)
            return out_points
        else:
            meandis=compare
            out_points=[[],[],[],[]]
            out_points[0]=walk_until_dis(x1,y1,slope12,meandis,1)
            out_points[1]=walk_until_dis(x2,y2,slope12,meandis,0)
            out_points[2]=walk_until_dis(x3,y3,slope12,meandis,0)
            out_points[3]=walk_until_dis(x4,y4,slope12,meandis,1)
            out_points=np.array(out_points)
            return out_points         
    #if fail to access, make a square
    else:
        meandis=compare/4
        out_points=[[],[],[],[]]
        slope13=(y1-y3)/(x1-x3)
        slope24=(y4-y2)/(x4-x2)
        out_points[0]=walk_until_dis(x1,y1,slope13,meandis,1)
        out_points[1]=walk_until_dis(x2,y2,slope24,meandis,0)
        out_points[2]=walk_until_dis(x3,y3,slope13,meandis,0)
        out_points[3]=walk_until_dis(x4,y4,slope24,meandis,1)
        out_points=np.array(out_points)
        return out_points

def make_big_rec(cont1:np.ndarray,
                 cont2:np.ndarray
                 )->tuple:
    '''
    vstack two conts into one cont and then getrec_info\n
    return tuple:  (center_x,center_y,wid,hei,rec_points,rec_area)
    
    '''
    big_cont=np.vstack((cont1,cont2))
    return getrec_info(big_cont)

def expand_rec_wid(rec_cont_list,expand_rate:float=1.5,img_size_yx:tuple=(1024,1280))->list:
    '''
    only expand short side
    return a list of expanded rec_conts
    '''
    out_list=[]
    dis=0
    for i in rec_cont_list:
        center_x,center_y,wid,hei,rec_points,_=getrec_info(i)
        out_points=[[],[],[],[]]
        x1,y1=rec_points[0]
        x2,y2=rec_points[1]
        x3,y3=rec_points[2]
        x4,y4=rec_points[3]
        hei=((x1-x2)**2+(y1-y2)**2)**0.5
        wid=((x2-x3)**2+(y2-y3)**2)**0.5
        if wid<hei:
            #always make side12 is short one
            wid,hei=hei,wid
            x1,x2,x3,x4=x2,x3,x4,x1
            y1,y2,y3,y4=y2,y3,y4,y1
            dis=hei*(expand_rate-1)/2
            #all is expanding in slope 12, the short side slope
            slope12=(y1-y2)/(x1-x2)
            #check if vertical
            if slope12==0 or math.isinf(slope12) or math.isnan(slope12):
                out_points[1]=[x1,y1-dis]
                out_points[2]=[x2,y2+dis]
                out_points[3]=[x3,y3+dis]
                out_points[0]=[x4,y4-dis]
                
            else:
                #1 and 4 is same, 1 and 2 is opposite
                
                        
                direc1='left' if slope12>0 else 'right'
                direc2='right' if slope12>0 else 'left'
                #make 1 is left down point
                out_points[1]=walk_until_dis(x1,y1,slope12,dis,direc1,img_size_yx=img_size_yx)
                out_points[2]=walk_until_dis(x2,y2,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[3]=walk_until_dis(x3,y3,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[0]=walk_until_dis(x4,y4,slope12,dis,direc1,img_size_yx=img_size_yx)
            out_list.append(np.array(out_points,dtype=np.int64))
        else:
            #always make side12 is short one
 
            dis=hei*(expand_rate-1)/2
            #all is expanding in slope 12, the short side slope
            
            slope12=(y1-y2)/(x1-x2)
            #check if vertical
            if slope12==0 or math.isnan(slope12) or math.isinf(slope12):
                out_points[0]=[x1,y1+dis]
                out_points[1]=[x2,y2-dis]
                out_points[2]=[x3,y3-dis]
                out_points[3]=[x4,y4+dis]
            else:
                #1 and 4 is same, 1 and 2 is opposite
                direc1='right' if slope12>0 else 'left'
                direc2='left' if slope12>0 else 'right'
                #make 1 is left down point
                out_points[0]=walk_until_dis(x1,y1,slope12,dis,direc1,img_size_yx=img_size_yx)
                out_points[1]=walk_until_dis(x2,y2,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[2]=walk_until_dis(x3,y3,slope12,dis,direc2,img_size_yx=img_size_yx)
                out_points[3]=walk_until_dis(x4,y4,slope12,dis,direc1,img_size_yx=img_size_yx)
            out_list.append(np.array(out_points,dtype=np.int64))
    return out_list     
        
            
        
        


def auto_search(ori_img:np.ndarray)->tuple:
    '''return 4 points that is the best cont of plate,tuple[0]is auto, tuple[1] is handmade'''
    img_edge_bin=get_edgebin2(ori_img)
    ori_bin=image_binary(ori_img)
    ori_conts,ori_arrs=cv2.findContours(img_edge_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    inner_conts=search_inner(ori_conts,ori_arrs,ori_bin)
    outer_conts=search_outer(ori_conts,ori_arrs,ori_bin)
    best_outer_num=0
    best_inner_num=0
    for i in range(len(inner_conts)):
        for j in range(len(outer_conts)):
            if isrelative(inner_conts[i],outer_conts[j],ori_conts,ori_arrs)==1:
                if iscenternear(inner_conts[i],outer_conts[j]):
                    best_outer_num=j
                    best_inner_num=i
    hand_points=make_plate(inner_conts[best_inner_num],img_edge_bin)
    auto_points=getrec_info(outer_conts[best_outer_num])[4]
    return auto_points,hand_points

def getdia_info(pt0,pt1,angle=0):
    '''input diagonal points of rec,\n
    return recinfo'''
    x0=pt0[0]
    x1=pt1[0]
    y0=pt0[1]
    y1=pt1[1]
    if angle==0:
        pt2=[x0,y1]
        pt3=[x1,y0]
        c_x,c_y,w,h,rec_cont,rec_area=getrec_info(np.array([pt0,pt1,pt2,pt3]))
        return c_x,c_y,w,h,rec_cont,rec_area
    
        