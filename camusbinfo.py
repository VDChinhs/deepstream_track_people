import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import pyds
from common.FPS import GETFPS
from common.bus_call import bus_call
from common.is_aarch_64 import is_aarch64
import cv2
import numpy as np
import landmarkcus
from utils.utility import is_image_blurry1, face_align, calculate_straight_score

padding_ratio_x = 1
padding_ratio_y = 0.5
face_threshold = 0.75
distance_eye_threshold = 6
straight_threshold = 13
blurry_threshold = 450
size_image_threshold = 170

tracker_id = set()
fps_streams = {}

MAX_ELEMENTS_IN_DISPLAY_META = 16
CONFIG_INFER = '/home/jetsonvy/DucChinh/config_infer_primary_face.txt'
STREAMMUX_WIDTH = 1280
STREAMMUX_HEIGHT = 720
DISPLAY_ON = True


def normal_landmark(landmarks, landmark_size):
    gain = min(landmark_size[0] / STREAMMUX_WIDTH,
            landmark_size[1] / STREAMMUX_HEIGHT)
    pad_x = (landmark_size[0] - STREAMMUX_WIDTH * gain) / 2.0
    pad_y = (landmark_size[1] - STREAMMUX_HEIGHT * gain) / 2.0

    normal_landmark_list = []
    for idx, (x, y) in enumerate(landmarks):
        x = int((x - pad_x) / gain) 
        y = int((y - pad_y) / gain)
        normal_landmark_list.append((x,y))
    
    return normal_landmark_list

def parse_face_from_meta(frame_meta, obj_meta):
    landmarks = landmarkcus.get_landmarks(obj_meta)[:-1]
    landmark_size = landmarkcus.get_landmarks(obj_meta)[-1]
    gain = min(landmark_size[0] / STREAMMUX_WIDTH,
            landmark_size[1] / STREAMMUX_HEIGHT)
    pad_x = (landmark_size[0] - STREAMMUX_WIDTH * gain) / 2.0
    pad_y = (landmark_size[1] - STREAMMUX_HEIGHT * gain) / 2.0

    batch_meta = frame_meta.base_meta.batch_meta
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    for idx, (x, y) in enumerate(landmarks):
        x = int((x - pad_x) / gain) 
        y = int((y - pad_y) / gain)

        if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        circle_params = display_meta.circle_params[display_meta.num_circles]
        circle_params.xc = x
        circle_params.yc = y
        circle_params.radius = 1
        circle_params.circle_color.red = 1.0
        circle_params.circle_color.green = 1.0
        circle_params.circle_color.blue = 1.0
        circle_params.circle_color.alpha = 1.0
        circle_params.has_bg_color = 1
        circle_params.bg_color.red = 0.0
        circle_params.bg_color.green = 0.0
        circle_params.bg_color.blue = 1.0
        circle_params.bg_color.alpha = 1.0
        display_meta.num_circles += 1

def set_custom_bbox(obj_meta, text_display):
    border_width = 0
    font_size = 20
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) - 100)))

    obj_meta.text_params.display_text = text_display
    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0
    obj_meta.text_params.font_params.font_name = 'Ubuntu'
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = x_offset
    obj_meta.text_params.y_offset = y_offset
    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0

def osd_sink_pad_buffer_probe(pad,info,u_data):
    global tracker_id

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number = frame_meta.frame_num
        current_index = frame_meta.source_id

        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGRA)

        print(f"Video {current_index} Frame number: {frame_number}")
        list_detection = []

        l_obj=frame_meta.obj_meta_list

        if frame_number < 5:
            l_frame = l_frame.next
            break

        frame_print = []

        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)

                if len(frame_print) != 0:
                    left = int(obj_meta.rect_params.left)
                    top = int(obj_meta.rect_params.top)
                    width = int(obj_meta.rect_params.width)
                    height = int(obj_meta.rect_params.height)
                    classid = obj_meta.class_id
                    landmark = landmarkcus.get_landmarks(obj_meta)[:-1]
                    masksize = landmarkcus.get_landmarks(obj_meta)[-1]
                    tracking_id = obj_meta.object_id
                    confidence = obj_meta.confidence
                    obj = {}
                    obj['classid'] = classid
                    obj['bbox'] = [left, top, left + width, top + height]
                    obj['confidence'] = confidence
                    obj['tracking_id'] = tracking_id
                    obj['landmark'] = landmarkcus.get_landmarks(obj_meta)[:-1]
                    obj['masksize'] = landmarkcus.get_landmarks(obj_meta)[-1]
                    list_detection.append(obj)
            except StopIteration:
                break

            tracking_id = obj_meta.object_id
            confidence = obj_meta.confidence
            
            left = int(obj_meta.rect_params.left)
            top = int(obj_meta.rect_params.top)
            width = int(obj_meta.rect_params.width)
            height = int(obj_meta.rect_params.height)
            classid = obj_meta.class_id
            landmark = landmarkcus.get_landmarks(obj_meta)[:-1]
            masksize = landmarkcus.get_landmarks(obj_meta)[-1]
            
            x1,y1,x2,y2 = left, top, left + width, top + height
        
            img_align = face_align(frame, np.array(normal_landmark(landmark, masksize), dtype=np.float32))
            var_blurry = is_image_blurry1(img_align)
            straight_score = calculate_straight_score(normal_landmark(landmark, masksize), (width, height))
            
            text_display = f"ID: {tracking_id}\n" \
                   f"Conf: {confidence:.2f}\n" \
                   f"Blurry: {var_blurry:.3f}\n"\
                   f"Straight: {straight_score:.4f}\n" #-0,002 đến 0.06
            
            parse_face_from_meta(frame_meta, obj_meta)
            set_custom_bbox(obj_meta, text_display)

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        if frame_number in frame_print and len(frame_print) != 0:
            print('Print Frame')
            for i in list_detection:
                x1,y1,x2,y2 = i['bbox']
                confidence = i['confidence']
                tracking_id = i['tracking_id']
                landmarks = i['landmark']
                landmark_size = i['masksize']
                
                for idx, (x, y) in enumerate(landmarks):
                    gain = min(landmark_size[0] / STREAMMUX_WIDTH,
                            landmark_size[1] / STREAMMUX_HEIGHT)
                    pad_x = (landmark_size[0] - STREAMMUX_WIDTH * gain) / 2.0
                    pad_y = (landmark_size[1] - STREAMMUX_HEIGHT * gain) / 2.0
                    x = int((x - pad_x) / gain) 
                    y = int((y - pad_y) / gain)
                    cv2.circle(frame, (x, y), radius=1, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f'{tracking_id}:{round(confidence, 2)}', (x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

            cv2.imwrite(f'/home/jetsonvy/DucChinh/frame{frame_number}.jpg', frame)

        fps_streams['stream{0}'.format(current_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	

def main(args):
    if len(args) != 2:
        sys.stderr.write("usage: %s <v4l2-device-path>\n" % args[0])
        sys.exit(1)

    GObject.threads_init()
    Gst.init(None)

    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating Source \n ")
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    stream_id = 0
    fps_streams['stream{0}'.format(stream_id)] = GETFPS(stream_id)
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")


    print("Creating Video Converter \n")

    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")

    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    tracker = Gst.ElementFactory.make("nvtracker", "nvtracker")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    if is_aarch64():
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing cam %s " %args[1])
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    source.set_property('device', args[1])
    streammux.set_property('width', STREAMMUX_WIDTH)
    streammux.set_property('height', STREAMMUX_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', CONFIG_INFER)
    sink.set_property('sync', False)

    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file',
                         '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
    tracker.set_property('display-tracking-id', 1)
    tracker.set_property('qos', 0)
    if tracker.find_property('enable_batch_process') is not None:
        tracker.set_property('enable_batch_process', 1)

    if tracker.find_property('enable_past_frame') is not None:
        tracker.set_property('enable_past_frame', 1)

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(tracker)
    pipeline.add(nvosd)
    pipeline.add(sink)
    if is_aarch64():
        pipeline.add(transform)

    print("Linking elements in the Pipeline \n")
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tracker)
    tracker.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))