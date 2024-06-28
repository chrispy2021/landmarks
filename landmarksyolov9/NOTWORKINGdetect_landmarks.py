import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, LoadStreamsOG
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


###MIJN CODE
import numpy as np

from V_SLAM_fcn.visual_tracker_fcn import *
from V_SLAM_fcn.ros_subscribers import *
from V_SLAM_fcn.spatial_filter_fcn import *

SHOW_LIVE_FEED = True


width = 640
height = 480

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


from openni import openni2
###MIJN CODE

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreamsOG(source, img_size=imgsz)#, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

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

    QUIT = False
    s = ""


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

    if READING_FROM_RECORDING:
        # make dataset equal to every dump in the save_stream.pckl file
        with open('save_stream.pckl', 'rb') as f:
            dataset = []
            while True:
                try:
                    dataset.append(pickle.load(f))
                except EOFError:
                    break


    for path, im, im0s, d_map, vid_cap in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)











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
                print(img_rgb_msg)
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
                    # im_rgb = cv2.circle(im_rgb, (int(obj_cent[0]), int(obj_cent[1])), 5, (0,255,0), -1)
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


            # if spatial_filter_text not in ['', ' ', None]:
                # print(spatial_filter_text)

            ### END LANDMARK CODE











            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov9-c-converted.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)