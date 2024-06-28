#!/usr/bin/env python
import cv2
import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *

import numpy as np

from V_SLAM_fcn.visual_tracker_fcn import *
from V_SLAM_fcn.ros_subscribers import *
from V_SLAM_fcn.spatial_filter_fcn import *
import parameters

SHOW_LIVE_FEED = True


width = 640
height = 480
#----------------------------------------------------------------------------------#

## ROS landmark publisher ---------------------------------------------------------#
import rospy
from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
from visualization_msgs.msg import MarkerArray
# from visualization_msgs.msg import Marker, MarkerArray
# from ROS_pub import *
TOPIC = '/v_landmarks'
## ROS landmark publisher node
rospy.init_node('landmark_publisher', anonymous=True)
lmk_pub = rospy.Publisher(TOPIC, LandmarkList, queue_size=1000) # 
#----------------------------------------------------------------------------------#
def log(*x, quit=True):
    print("--> ", x)
    if quit:
        exit(0)

def olog(*x, prob=0.01): # occasional log
    from random import random as rd_
    if rd_() < prob:
        log(x, quit=False)

def detect(save_img=False):
    ## Writing output video
    fps = 30
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output_path = './name_video.avi'
    out_vid_write = cv2.VideoWriter(output_path, codec, fps, (width, height))

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    import time
    START_TIME = time.time()


    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    # google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()

    model.to(device).eval()
    if half:
        model.half()  # to FP16
        
    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('./code/Landmarks_vSLAM/weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # for yolov5 v1.0 yolov5s.pt this is a list of a few classes [person, tv, keyboard, mouse, etc]
    names = model.module.names if hasattr(model, 'module') else model.names
    fps_disp = 0.0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    

    ## Init
    first_detection = True
    lmk_id = 0
    old_objects = {}
    old_num = 0
    codebook_match = {}
    txt = " "
    correct_hist = {}
    rbt_glb_pose = robot_global_pose() ## robot global pose ### EMPTY
    lmk_global = landmarks_global_pose() ## landmarks global pose ### EMPTY
    lmk_obsv_poses = {}
    spatial_filter_text = ' '
    frames_wo_det = 0

    MAKE_RANDOM_SCREENSHOTS = True

    RECORDING_MODE = False
    READING_FROM_RECORDING = True

    # both false means live from camera

    if RECORDING_MODE and READING_FROM_RECORDING:
        # throw error
        raise ValueError("RECORDING_MODE and READING_FROM_RECORDING cannot be True at the same time")


    import pickle

    if RECORDING_MODE:
        with open('save_stream.pckl', 'wb') as f:
            pass

    screenshot_idx = 0
    count = 0

    START_LOOP_TIME = time.time()
    if READING_FROM_RECORDING:
        # make dataset equal to every dump in the save_stream.pckl file
        # with open('small_round.pckl', 'rb') as f:
        with open('big_round_lab42.pckl', 'rb') as f:
            # dataset = []
            # ctr = 0
            # while True:
            #     ctr += 1
                    
            #     try:
            #         dataset.append(pickle.load(f))
            #         if ctr <= 1500:
            #             dataset = []
            #     except EOFError:
            #         break
            #     if ctr > 3000:
            #         break

            dataset = []
            ctr = 0
            final = False
            # run the for loop in pickle.load chunks of size 1500
            while True:

                while True:
                    try:
                        dataset.append(pickle.load(f))
                        ctr += 1
                        if ctr % 1500 == 0:
                            break
                    except EOFError:
                        final = True
                        break
                
                print(len(dataset))

                ## Main loop
                for path, img, im0s, d_map, vid_cap in dataset: # path = [0], vid_cap = None, d_map depth map, im0s lijkt de camera output, img snap ik niet
                    # dataset.mode == 'images'
                    count += 1
                    # im0s[0] = cv2.cvtColor(im0s[0], cv2.COLOR_BGR2RGB)
                    # im[0] = cv2.cvtColor(im[0], cv2.COLOR_BGR2RGB)

                    if RECORDING_MODE:
                        with open('save_stream.pckl', 'ab') as f:
                            # pickle dump path, im, im0s, d_map and vid_cap
                            pickle.dump((path, img, im0s, d_map, vid_cap), f)

                    img[0] = img[0][::-1, :, :]
                


                    img = torch.from_numpy(img).to(device) # to tensor
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0

                    dmap = d_map
                    if dmap is not None:
                        d4d = np.uint8(dmap.astype(float) * 255 / 2 ** 12 - 1)  # Correct the range. Depth images are 12bits
                        d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
                        timestamp = rospy.Time.now()
                        rbt_glb_pose.trans_mat()
                        lmk_global.lmk_gPoses_list()

                    # depth maps, dmap is basically black, probably because alll the values are high and normalization isnt applied
                    # d4d shows a semi proper depth map, does not seem to be perfect but i assume this isnt a problem
                    # cv2.imshow('d', dmap) # * 255)
                    # cv2.imshow('4', d4d)
                    # log(dmap.shape)
                    # print(np.random.choice(d4d.flatten()))
                    # print(np.max(dmap))

                    # fr = img[0].cpu().numpy().transpose(1, 2, 0) # img is a tensor, fr is a numpy array
                    # cv2.imshow('i', fr) # this is the image with the borders on top and bottom
                    # cv2.imshow('i', np.array(img[0].))
                    
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0) #weird values? only 0.44702 seems to show up
                    # i dont get this
                    # cv2.imshow('i',img[0])
                    # shape torch.Size([1, 3, 352, 416]), misschien later
                    # print(np.array(img[0][0].cpu().numpy()))

                    # cv2.imshow('i', np.array(img[0][0].cpu().numpy(), dtype=np.float)) r
                    # cv2.imshow('j', np.array(img[0][1].cpu().numpy(), dtype=np.float)) g
                    # cv2.imshow('k', np.array(img[0][2].cpu().numpy(), dtype=np.float)) b
                    # this shows in greyscale

                    # img is a weird rbg channel image version with borders on top and bottom of image

                    ## Online operation data
                    online_data = {'timestamp': timestamp,
                                'depth_map': dmap, # vgm showt dit raar omdat dus alle values tussen 0 en 1 zitten maar hij pakt 0-255 greyscale bij imshow
                                'robot_global_pose': rbt_glb_pose.trans} # deze verandert gwn niet

                    # Inference
                    t1 = torch_utils.time_synchronized()
                    pred = model(img, augment=opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    # if pred[0] is not None:
                    #     print(pred[0].shape) # torch.Size([n, 6]) bij n object detetctions met n > 0

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        if webcam:  # batch_size >= 1 # webcam is alawys true
                            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy() # p = 0 , s = '0: '
                        else:
                            p, s, im0 = path, '', im0s

                        # if i != 0:
                        #     print(i) # pred is altijd len(pred) == 1, de detections zitten in pred[0]

                        im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

                        save_path = str(Path(out) / Path(p).name) # inferences/output/0 ?????????
                        s += '%gx%g ' % img.shape[2:]  # print string, image nxm size
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                        
                        # print(len(det) # if det is not None else 'None')
                        if det is not None and len(det): # if at least 1 detection
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += '%g %ss, ' % (n, names[int(c)])  # add to string
                                # print(n,s) # n is niet echt iets, voorbeeld output s: 0: 352x416 3 tvs, 1 mouses
                                
                            # Write results
                            for *xyxy, conf, cls in det:
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                        file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                                if save_img or view_img:  # Add bbox to image
                                    label = '%s %.2f' % (names[int(cls)], conf) # get class name, confidence score
                                    #plot_one_box(xyxy, im_rgb, label=label, color=colors[int(cls)], line_thickness=3)
                                    plot_one_box(xyxy, im_rgb, label=label, color=(0,255,0), line_thickness=3)


                        ### END DRAW BOXES ON SCREEN


















                    ### publish odom start

                        from sensor_msgs.msg import Image
                        from cv_bridge import CvBridge
                        bridge = CvBridge()

                        ## Create the publisher
                        pub_rgb = rospy.Publisher('/camera/color/image_rect_color', Image, queue_size=10)
                        pub_depth = rospy.Publisher('/camera/depth/image_rect_raw', Image, queue_size=10)
                        # rospy.init_node('camera', anonymous=True)
                        # rate = rospy.Rate(30)  # 30hz

                        if not READING_FROM_RECORDING: # if live feed
                            rgb_stream = dataset.rgb_stream
                            depth_stream = dataset.depth_stream

                            ## Get the rgb frame
                            frame_rgb = rgb_stream.read_frame()
                            frame_data_rgb = frame_rgb.get_buffer_as_uint8()
                            img_rgb = np.frombuffer(frame_data_rgb, dtype=np.uint8)
                            img_rgb.shape = (480, 640, 3)
                            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                            img_rgb_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
                            # print(img_rgb_msg)
                            pub_rgb.publish(img_rgb_msg)

                            ## Get the depth frame
                            frame_depth = depth_stream.read_frame()
                            frame_data_depth = frame_depth.get_buffer_as_uint16()
                            img_depth = np.frombuffer(frame_data_depth, dtype=np.uint16)
                            img_depth.shape = (480, 640)
                            img_depth = img_depth.astype(np.float32)
                            img_depth = cv2.normalize(img_depth, None, 0, 255, cv2.NORM_MINMAX)
                            img_depth = img_depth.astype(np.uint8)
                            img_depth_msg = bridge.cv2_to_imgmsg(img_depth, encoding="mono8")
                            pub_depth.publish(img_depth_msg)

                        else: # if reading from recording
                            img_rgb = im0s[0]
                            img_rgb_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
                            pub_rgb.publish(img_rgb_msg)

                            img_depth = d4d
                            img_depth_msg = bridge.cv2_to_imgmsg(img_depth, encoding="8UC3")
                            pub_depth.publish(img_depth_msg)


                        from nav_msgs.msg import Odometry
                        odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)

                        # listen to /orb_slam2_rgbd/pose

                        def callback(data):
                            

                            # odom message
                            # print(data, type(data))
                            # print(data.pose)
                            odom = Odometry()
                            # print(odom)

                            odom.header.stamp = rospy.Time.now()
                            odom.header.frame_id = "odom"
                            odom.child_frame_id = "base_link"

                            # odom.pose.pose = data.pose

                            odom.pose.pose.position.x = data.pose.position.x / 10
                            odom.pose.pose.position.y = data.pose.position.y / 10
                            odom.pose.pose.position.z = 0

                            odom.pose.pose.orientation.x = data.pose.orientation.x
                            odom.pose.pose.orientation.y = data.pose.orientation.y
                            odom.pose.pose.orientation.z = data.pose.orientation.z
                            odom.pose.pose.orientation.w = data.pose.orientation.w

                            odom.twist.twist.linear.x = 0
                            odom.twist.twist.linear.y = 0
                            odom.twist.twist.linear.z = 0
                            odom.twist.twist.angular.x = 0
                            odom.twist.twist.angular.y = 0
                            odom.twist.twist.angular.z = 0

                            # print("odom ",odom.pose.pose.position.x,' ', odom.pose.pose.position.y)
                            odom_pub.publish(odom)



                        rospy.Subscriber("/orb_slam2_rgbd/pose", PoseStamped, callback)

                        ### publish odom end
















                        ### START LANDMARK CODE

                        ## Check if any detections happen
                        detections = pred[0]
                        objects = detections

                        # print("objects_1: ", objects)
                        # print(det == objects) # always true
                        # det = pred[0] = detections = objects = nx6 tensor met detections, x1, y1, x2, y2, conf, class
                        if det is not None:
                            ## number of detections:
                            num_d = len(detections)
                            for i in range(num_d):
                                det_info = detections[i].data.cpu().tolist()
                                bbox = [int(j) for j in det_info[0:4]]  ## bbox = [x1 y1 x2 y2]

                                ## Check if the object is within range
                                obj_cent = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]  ## [x y] center pixel of the area the object is located
                                ## Optimal operation range: 0.8-3.5m
                                im_rgb = cv2.circle(im_rgb, (int(obj_cent[0]), int(obj_cent[1])), 5, (0,255,0), -1)
                                d4d = cv2.circle(d4d, (int(obj_cent[0]), int(obj_cent[1])), 5, (0,0,255), -1)
                                # cv2.imshow('i', dmap * 255) # if there are detections, place a dot on the center of object on the depth map
                                # dmap is de gekke met een hoge range, d4d is genormaliseerd 0-255
                                # print(dmap[int(obj_cent[1]), int(obj_cent[0])])
                                if dmap[int(obj_cent[1]), int(obj_cent[0])] < 800 or dmap[int(obj_cent[1]), int(obj_cent[0])] > 3500: # wtf?? are these values
                                    objects = torch.cat([detections[0:i], detections[i+1:]]) # huh?
                                    # poging tot uitleg: de laatste detection die onder de condition valt, vervalt

                                    # print(objects.shape[0]) # == num of detections -1  
                                    # print("objects_2: ", objects, end='\n\n\n')
                                    # print(detections[0:0]) # this is nothing (0,6) tensor
                                    # if objects.shape[0] > 0:
                                    #     print(detections, objects)

                            if not objects.cpu().tolist(): objects = None
                        
                        # i dont get this weird filter but if there is only one object detected, and the center is not between 800-3500, then objects is None

                        ## Track and sample landmarks
                        if objects is not None:
                            
                            if first_detection: # the very very very first detection, true only once in runtime
                                frames_wo_det = 0
                                lmk_id += 1 # dit wordt in init op 0 gezet dus lmk_id = 1
                                old_num = len(objects) # aantal detections in objects dus
                                
                                # cv2.imshow('a', im0) # screenshot degene zonder labels bboxes, is ook iets donkerder in kleur?
                                # cv2.imshow('b', im_rgb) # screenshot met bboxers, iets lichter in kleur
                                # cv2.imshow('c', online_data['depth_map']) # dmap, de donkere versie, jeziet hier niet echt iets op

                                # online data[robot_global_pose] = []
                                im_rgb, old_objects, tracked_histograms,  lmkObsv = new_landmarks(objects, lmk_id, im0, im_rgb, online_data, names)
                                first_detection = False
                                codebook_match = {}
                                correct_hist = {}
                            else:
                                ## Track objects and serch for best match in keyframes
                                tracker = track_detections(old_num,  im0, im_rgb, old_objects, objects, lmk_id, tracked_histograms, codebook_match, correct_hist, online_data, lmkObsv, names, frames_wo_det)
                                lmk_id = tracker.id
                                old_objects = tracker.old_objects
                                old_num = tracker.old_num
                                im_rgb = tracker.disp
                                tracked_histograms = tracker.tracked_histograms
                                codebook_match = tracker.codebook_match
                                correct_hist = tracker.keyframes_hist
                                lmkObsv = tracker.lmkObsv
                                frames_wo_det = 0

                                if tracker.publish_flag:
                                    # print(lmk_pub)
                                    ## Filter spatially and publish correct landmarks to cartographer
                                    lmk_gp = lmk_global.landmarks_global_poses ## Global poses of landmarks
                                    filtered_obsv = spatial_filter(lmkObsv, correct_hist, tracked_histograms, lmk_gp, lmk_obsv_poses, spatial_filter_text, lmk_pub, lmk_id)

                                    correct_hist = filtered_obsv.keyframes_copy
                                    tracked_histograms = filtered_obsv.tracked_hists
                                    lmkObsv = filtered_obsv.lmkObsv
                                    spatial_filter_text = filtered_obsv.spatial_filter_text
                                    lmk_id = filtered_obsv.lmk_id

                                if hasattr(tracker, 'sampler_txt')  and tracker.sampler_txt != " ":
                                    txt = tracker.sampler_txt

                        else: ## no detections
                            frames_wo_det +=1




                        ### END LANDMARK CODE








                            
                    #show text, fps display
                    img2 = cv2.putText(im_rgb, txt, (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    # if txt != " ":
                    #     # print("main1: ",txt)
                    img2 = cv2.putText(im_rgb, spatial_filter_text, (0, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    # if spatial_filter_text != " ":
                    #     print("main2: ", spatial_filter_text)
                    fps_disp = (fps_disp + (1. / (torch_utils.time_synchronized() - t1))) / 2
                    img2 = cv2.putText(img2, "FPS: {:.2f}".format(fps_disp), (0, 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                    # Stream results
                    # print(view_img, save_img)#, img2)

                    if view_img or save_img:
                        out_vid_write.write(img2)
                        
                        if (SHOW_LIVE_FEED):
                            cv2.imshow("image", img2)

                            if MAKE_RANDOM_SCREENSHOTS:
                                screenshot_idx += 1
                                # save image as png
                                rand_int = np.random.randint(0, 1297)
                                if rand_int < 100:
                                    cv2.imwrite(f"LAB42DATASET/YOLOV1SCREENSHOTSLAB42/pic{screenshot_idx}.png", img2)

                        else:
                            cv2.imshow("image", np.zeros(3))
                            
                        if cv2.waitKey(1) & 0xFF == ord('q'):#cv2.waitKey(1) == ord('q'):  # q to quit
                            cv2.destroyAllWindows()
                            break

                    # Save results (image with detections)
                    save_img = False
                    if save_img:
                        # print(dataset.mode) dat is altijd zo
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, display)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))

                dataset = []
                if final:
                    break


    END_TIME = time.time()

    print(START_TIME, START_LOOP_TIME, END_TIME)
    print(START_LOOP_TIME - START_TIME, END_TIME - START_LOOP_TIME)

    # Field dataset
    # 1718874968.8903718 1718874976.2051797 1718875116.427008
    # 7.314807891845703 140.22182822227478

    # real field
    # 1718883471.2463713 1718883478.56681 1718883620.472492
    # 7.320438623428345 141.90568208694458

    print("Frames in dataset: ", count)
    # 1297 frames in veld dataset
    
    # lab42 dataset:
#     1719399533.4338045 1719399533.5725734 1719399813.5102725
# 0.13876891136169434 279.93769907951355
    

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./LANDMARKS/landmarksyolov5v1/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.85, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()