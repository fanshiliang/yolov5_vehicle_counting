import numpy as np

import tracker
from detector import Detector
import cv2

# 初始化2个撞线polygon
list_pts_blue = []
list_pts_yellow = []
# list 与蓝色polygon重叠
list_overlapping_blue_polygon = []

# list 与黄色polygon重叠
list_overlapping_yellow_polygon = []

color_polygons_image = None

drag_start = None
sel = None

def onmouse(event, x, y, flags, param):
    global drag_start
    global sel
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = (0,0,0,0)
    elif event == cv2.EVENT_LBUTTONUP:
        if sel[2] > sel[0] and sel[3] > sel[1]:
            param[1].append([sel[0], sel[1]])
            param[1].append([sel[0],sel[3]])
            param[1].append([sel[2], sel[3]])
            param[1].append([sel[2], sel[1]])
        drag_start = None
    elif drag_start:
        #print flags
        if flags & cv2.EVENT_FLAG_LBUTTON:
            minpos = min(drag_start[0], x), min(drag_start[1], y)
            maxpos = max(drag_start[0], x), max(drag_start[1], y)
            sel = (minpos[0], minpos[1], maxpos[0], maxpos[1])
            img = param[0].copy()
            cv2.rectangle(img, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 1)
            cv2.imshow("select_line", img)
        else:
            print("selection is complete")
            drag_start = None

def recalculate_coordinate(list_pts, video_size, target_size):
    print(video_size)
    v_h = video_size[0]
    v_w = video_size[1]
    t_h = target_size[0]
    t_w = target_size[1]
    print(v_h, v_w, t_h, t_w)
    for pts in list_pts:
        pts[0] = int(pts[0] * t_w / v_w)
        pts[1] = int(pts[1] * t_h / v_h)
    return list_pts

if __name__ == '__main__':

    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = None
    color_polygons_image = None
    polygon_blue_value_1 = None
    polygon_yellow_value_2 = None
    # 初始化2个撞线polygon
    # list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
    #                  [299, 375], [267, 289]]

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('./video/路口1.mp4')

    first_line_initialized = False
    second_line_initialized = False
    first_iter = True
    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        if not first_line_initialized:
            cv2.namedWindow("select_line",1)
            cv2.setMouseCallback("select_line", onmouse, (im, list_pts_blue))
            cv2.imshow('select_line', im)
            cv2.waitKey(0)
            cv2.destroyWindow('select_line')
            mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
            print(list_pts_blue)
            list_pts_blue = recalculate_coordinate(list_pts_blue, (im.shape[0], im.shape[1]), (mask_image_temp.shape[0], mask_image_temp.shape[1]))
            print(list_pts_blue)
            ndarray_pts_blue = np.array(list_pts_blue, np.int32)
            polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
            polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]
            
            first_line_initialized = True

        if not second_line_initialized:
            cv2.namedWindow("select_line",1)
            cv2.setMouseCallback("select_line", onmouse, (im, list_pts_yellow))
            cv2.imshow('select_line', im)
            cv2.waitKey(0)
            cv2.destroyWindow('select_line')
            mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
            print(list_pts_yellow)
            list_pts_yellow = recalculate_coordinate(list_pts_yellow, (im.shape[0], im.shape[1]), (mask_image_temp.shape[0], mask_image_temp.shape[1]))
            print(list_pts_yellow)
            ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
            polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
            polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

            second_line_initialized = True
        
           
        if first_iter:
            polygon_mask_blue_and_yellow = polygon_yellow_value_2 + polygon_blue_value_1
            polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))
            
            # 蓝 色盘 b,g,r
            blue_color_plate = [255, 0, 0]
            # 蓝 polygon图片G
            blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
            # 黄 色盘`
            #  `
            yellow_color_plate = [0, 255, 255]
            # 黄 polygon图片
            yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

            # 彩色图片（值范围 0-255）
            # color_polygons_image = blue_image + yellow_image
            color_polygons_image = yellow_image + blue_image
            # 缩小尺寸，1920x1080->960x540
            color_polygons_image = cv2.resize(color_polygons_image, (960, 540))
            first_iter = False

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=1)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1

                        print(f'类别: {label} | id: {track_id} | 撞线 | 撞线总数: {up_count} | id列表: {list_overlapping_yellow_polygon}')

                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1

                        print(f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass

        text_draw = 'traffic: ' + str(down_count)
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

        pass
    pass

    capture.release()
    cv2.destroyAllWindows()
