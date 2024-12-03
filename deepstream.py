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
import uuid
import landmarkcus
from utils.utility import get_day, get_time, distance, find_center_of_quadrilateral, is_image_blurry1, distance_eye
from Kafka.producer import send_image

padding_ratio_x = 1
padding_ratio_y = 0.5
face_threshold = 0.75
distance_eye_threshold = 6
straight_threshold = 13
blurry_threshold = 300
size_image_threshold = 170

tracker_id = set()
fps_streams = {}

MAX_ELEMENTS_IN_DISPLAY_META = 16
CONFIG_INFER = '/home/jetsonvy/DucChinh/config_infer_primary_face.txt'
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0
DISPLAY_ON = True
SOURCE_LIST = {
    # 0: 'file:////home/jetsonvy/DucChinh/videos/People_Walking1.mp4',
    1: 'file:////home/jetsonvy/DucChinh/videos/track.mp4',
    # 0: 'rtsp://192.168.102.6:8880/h264.sdp',
}


def main():
    Gst.init(None)
    loop = GObject.MainLoop()
    pipeline = Gst.Pipeline()

    streammux = make_elm_or_print_err("nvstreammux", "nvstreammux")

    for key, value in SOURCE_LIST.items():
        source = create_uridecode_bin(key, value, streammux)
        pipeline.add(source)

    converter = make_elm_or_print_err("nvvideoconvert", "nvvideoconvert")
    filter = make_elm_or_print_err("capsfilter", "capsfilter")
    pgie = make_elm_or_print_err("nvinfer", "pgie")
    tracker = make_elm_or_print_err("nvtracker", "nvtracker")
    if DISPLAY_ON:
        sinkdp = make_elm_or_print_err('nv3dsink', 'nv3dsink')
        osd = make_elm_or_print_err("nvdsosd", "nvdsosd")
    else:
        sink = make_elm_or_print_err("fakesink", "fakesink")

    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('width', STREAMMUX_WIDTH)
    streammux.set_property('height', STREAMMUX_HEIGHT)
    streammux.set_property('enable-padding', 0)
    streammux.set_property('live-source', 1)
    streammux.set_property('attach-sys-ts', 1)

    filter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    pgie.set_property('config-file-path', CONFIG_INFER)
    pgie.set_property('qos', 0) # Cố gắng xử lý hết khung hình

    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file',
                         '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
    tracker.set_property('display-tracking-id', 1)
    tracker.set_property('qos', 0)

    if DISPLAY_ON:
        osd.set_property('process-mode', int(pyds.MODE_GPU))
        osd.set_property('qos', 0)
        sinkdp.set_property("sync", False)
        sinkdp.set_property("max-lateness", -1)
        sinkdp.set_property("qos", 0)
    else:
        sink.set_property("sync", False)
        sink.set_property("max-lateness", -1)
        sink.set_property("qos", 0)


    if tracker.find_property('enable_batch_process') is not None:
        tracker.set_property('enable_batch_process', 1)

    if tracker.find_property('enable_past_frame') is not None:
        tracker.set_property('enable_past_frame', 1)

    pipeline.add(streammux)
    pipeline.add(converter)
    pipeline.add(filter)
    pipeline.add(pgie)
    pipeline.add(tracker)
    if DISPLAY_ON:
        pipeline.add(osd)
        pipeline.add(sinkdp)
    else:
        pipeline.add(sink)


    streammux.link(converter)
    converter.link(filter)
    filter.link(pgie)
    pgie.link(tracker)
    if DISPLAY_ON:
        tracker.link(osd)
        osd.link(sinkdp)
    else:
        tracker.link(sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    gie_src_pad = tracker.get_static_pad('src')
    if not gie_src_pad:
        sys.stderr.write("Unable to get src pad \n")
    else:
        gie_src_pad.add_probe(Gst.PadProbeType.BUFFER, gie_probe, 0)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

def make_elm_or_print_err(factoryname, name):
    """
    Tạo phần tử GStreamer hoặc in lỗi nếu không thành công.
    """
    print("Creating", factoryname)
    element = Gst.ElementFactory.make(factoryname, name)
    if not element:
        sys.stderr.write(f"Không thể tạo {factoryname}\n")
        sys.exit(1)
    return element
    
def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find('decodebin') != -1:
        Object.connect('child-added', decodebin_child_added, user_data)
    if name.find('nvv4l2decoder') != -1:
        Object.set_property('drop-frame-interval', 0)
        Object.set_property('num-extra-surfaces', 1)
        if is_aarch64():
            Object.set_property('enable-max-performance', 1)
        else:
            Object.set_property('cudadec-memtype', 0)
            Object.set_property('gpu-id', GPU_ID)

def cb_newpad(decodebin, pad, user_data):
    streammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)
    if name.find('video') != -1:
        if features.contains('memory:NVMM'):
            if pad.link(streammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write('ERROR: Failed to link source to streammux sink pad\n')
        else:
            sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin')

def create_uridecode_bin(stream_id, uri, streammux):
    bin_name = 'source-bin-%04d' % stream_id
    # bin = make_elm_or_print_err('uridecodebin', bin_name)
    if uri == "/dev/video0":
        bin = make_elm_or_print_err("v4l2src", bin_name)
        bin.set_property('device', uri)
    else:
        bin = make_elm_or_print_err('uridecodebin', bin_name)
        bin.set_property('uri', uri)
    # bin.set_property('uri', uri)
    pad_name = 'sink_%u' % stream_id
    streammux_sink_pad = streammux.get_request_pad(pad_name)
    bin.connect('pad-added', cb_newpad, streammux_sink_pad)
    bin.connect('child-added', decodebin_child_added, 0)
    fps_streams['stream{0}'.format(stream_id)] = GETFPS(stream_id)
    return bin

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
        circle_params.radius = 3
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

def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))

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

def gie_probe(pad, info, user_data):
    global tracker_id

    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        current_index = frame_meta.source_id

        n_frame = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
        frame = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGRA)

        print(f"Video {current_index} Frame number: {frame_number}")
        list_detection = []

        l_obj = frame_meta.obj_meta_list
        
        if frame_number < 5:
            l_frame = l_frame.next
            break

        frame_print = []
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

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
            if DISPLAY_ON:
                parse_face_from_meta(frame_meta, obj_meta)
                set_custom_bbox(obj_meta)

            tracking_id = obj_meta.object_id
            confidence = obj_meta.confidence
            if confidence < face_threshold:
                l_obj = l_obj.next
                continue

            if tracking_id in tracker_id and tracking_id is not None:
                # print(f'Id: {tracking_id} đã ồn tại')
                l_obj = l_obj.next
                continue

            left = int(obj_meta.rect_params.left)
            top = int(obj_meta.rect_params.top)
            width = int(obj_meta.rect_params.width)
            height = int(obj_meta.rect_params.height)
            classid = obj_meta.class_id
            landmark = landmarkcus.get_landmarks(obj_meta)[:-1]
            masksize = landmarkcus.get_landmarks(obj_meta)[-1]

            center = find_center_of_quadrilateral(landmark[0], landmark[3], landmark[4], landmark[1])
            current_straight = distance(center, landmark[2])
            if current_straight > straight_threshold:
                # print(f'Mặt không thẳng: {current_straight} > {straight_threshold}')
                l_obj = l_obj.next
                continue

            current_distance_eye = distance_eye(landmark)
            if current_distance_eye < distance_eye_threshold:
                # print('Mặt ngang quá', current_distance_eye)
                l_obj = l_obj.next
                continue
            
            x1,y1,x2,y2 = left, top, left + width, top + height
            padding_x = int((x2 - x1) * padding_ratio_x)
            padding_y = int((y2 - y1) * padding_ratio_y)
            if not (int(y1) - padding_y > 0 and int(y2) + padding_y < STREAMMUX_HEIGHT and int(x1) - padding_x > 0 and int(x2) + padding_x < STREAMMUX_WIDTH):
                # print(f'Cắt ảnh vượt quá padding: {padding_x}x{padding_y}')
                l_obj = l_obj.next
                continue              

            img_face = frame[int(y1) - padding_y : int(y2) + padding_y, int(x1) - padding_x : int(x2) + padding_x]
            if not (img_face.shape[0] > size_image_threshold and img_face.shape[1] > size_image_threshold):
                # print(f'Ảnh quá nhỏ: {img_face.shape[0]}x{img_face.shape[1]} < {size_image_threshold}x{size_image_threshold}')
                l_obj = l_obj.next
                continue
            
            var_blurry = is_image_blurry1(img_face)
            if var_blurry < blurry_threshold:
                # print(f'Ảnh quá mờ: {var_blurry} < {blurry_threshold}')
                l_obj = l_obj.next
                continue

            current_time = get_time()
            today = get_day()

            name_file = str(uuid.uuid4()).replace('-', '')
            # cv2.imwrite(f'/home/jetsonvy/DucChinh/images/{tracking_id}_{name_file}.jpg', img_face)
            send_image(img_face, current_index, "entry")

            tracker_id.add(tracking_id)
            print(f'Tên: {tracking_id}, Score: {confidence}, Time: {current_time}, Straight: {current_straight}, Blurry: {var_blurry}')

            try:
                l_obj = l_obj.next
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
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

if __name__ == '__main__':
    main()
