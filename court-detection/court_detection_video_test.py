import os
import time
from scipy import signal
import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from court_detection import CourtDetector
from ball_detection import BallDetector

def calculate_ball_positions(court_detector, ball_positions):
    """
    Calculate the feet position of both players using the inverse transformation of the court and the boxes
    of both players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    inv_mats = court_detector.game_warp_matrix
    positions_1 = []

    # Bottom player feet locations
    for i, box in enumerate(ball_positions):
        if box[0] is not None:
            feet_pos = np.array([box[0].item(), box[1].item()]).reshape((1, 1, 2))
            feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
            positions_1.append(feet_court_pos)

    # Smooth both feet locations
    positions_1 = np.array(positions_1)
    smoothed_1 = np.zeros_like(positions_1)
    smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
    smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)

    return smoothed_1

def find_ball_position(ball_positions, verbose=1):
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    # ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    # ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    # LEFT OFF HERE, NEED TO USE THESE BALL COORDINATES AND THROW OUT ALL THE PLAYER STUFF 
    ball_f2_y = interp1d(x, y1, fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()
    # return np.array([ball_f2_x, ball_f2_y]) #THIS IS AN ARRAY OF THE BALL POSITIONS IN THE ENTIRE THING...I THINK 
    return np.array([smooth_x, smooth_y]) #THIS IS AN ARRAY OF THE BALL POSITIONS IN THE ENTIRE THING...I THINK 

def create_top_view(court_detector, ball_detector):
    """
    Creates top view video of the gameplay
    """
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('/Users/tyler/Documents/GitHub/basketballVideoAnalysis/court_detect_output/top_view.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
    # ball location on court
    # smoothed_1, smoothed_2 = find_ball_position(ball_positions=ball_detector.xy_coordinates)
    smoothed_1 = calculate_ball_positions(court_detector=court_detector, ball_positions=ball_detector.xy_coordinates)

    for ball_pos_1 in zip(smoothed_1):
        frame = court.copy()
        frame = cv2.circle(frame, (int(ball_pos_1[0]), int(ball_pos_1[1])), 10, (0, 0, 255), 15)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()

def detect_court():
    videoin = "/Users/tyler/Documents/GitHub/basketballVideoAnalysis/shortQatar.mp4"
    video = cv2.VideoCapture(videoin)




    # initialize extractors
    court_detector = CourtDetector()
    ball_detector = BallDetector('/Users/tyler/Documents/GitHub/basketballVideoAnalysis/court-detection/tracknet_weights_2_classes.pth', out_channels=2)
    # detection_model = DetectionModel(dtype=dtype)
    # pose_extractor = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    # stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    # ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)

    # Load videos from videos path
    # video = cv2.VideoCapture("/Users/tyler/Documents/GitHub/basketballVideoAnalysis/tennis_rally.mp4")

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    # Loop over all frames in the videos
    while True:
        start_time = time.time()

        ret, frame = video.read()
        frame_i += 1

        if ret:
            if frame_i == 1:
                court_detector.detect(frame)
                print(f'Court detection {"Success" if court_detector.success_flag else "Failed"}')
                print(f'Time to detect court :  {time.time() - start_time} seconds')
                start_time = time.time()

            # cv2.imshow('court test', frame)
            # img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
            court_detector.track_court(frame)
            # detect
            # detection_model.detect_player_1(frame.copy(), court_detector)
            # detection_model.detect_top_persons(frame, court_detector, frame_i)

            # Create stick man figure (pose detection)
            # if stickman:
            #     pose_extractor.extract_pose(frame, detection_model.player_1_boxes)

            ball_detector.detect_ball(court_detector.delete_extra_parts(frame))

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            # if not frame_i % 100:
            #     print('')
        else:
            break
    # print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')
    video.release()
    cv2.destroyAllWindows()

    # create_top_view(court_detector, ball_detector)

    add_data_to_video(input_video=videoin, court_detector=court_detector, show_video=True, with_frame=1, output_folder='/Users/tyler/Documents/GitHub/basketballVideoAnalysis/court_detect_output', output_file='output', ball_detector=ball_detector)



def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height

def add_data_to_video(input_video, court_detector, show_video, with_frame, output_folder, output_file, ball_detector):
    """
    Creates new videos with pose stickman, face landmarks and blinks counter
    :param input_video: str, path to the input videos
    :param df: DataFrame, data of the pose stickman positions
    :param show_video: bool, display output videos while processing
    :param with_frame: int, output videos includes the original frame with the landmarks
    (0 - only landmarks, 1 - original frame with landmarks, 2 - original frame with landmarks and only
    landmarks (side by side))
    :param output_folder: str, path to output folder
    :param output_file: str, name of the output file
    :return: None
    """

    # Read videos file
    cap = cv2.VideoCapture(input_video)

    # videos properties
    fps, length, width, height = get_video_properties(cap)

    final_width = width * 2 if with_frame == 2 else width

    # Video writer
    # add the fps to make the video name unique 
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + f"_{fps}" + '.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (final_width, height))

    # initialize frame counters
    frame_number = 0
    orig_frame = 0
    while True:
        orig_frame += 1
        print('Creating new videos frame %d/%d  ' % (orig_frame, length), '\r', end='')
        if not orig_frame % 100:
            print('')
        ret, img = cap.read()

        if not ret:
            break
        cv2.putText(img=img, text="TEST", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,255), org=(50, 50))
        # initialize frame for landmarks only
        img_no_frame = np.ones_like(img) * 255

        # add Court location
        img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
        img_no_frame = court_detector.add_court_overlay(img_no_frame, overlay_color=(0, 0, 255), frame_num=frame_number)
    # top_baseline [676.44324, 255.8075, 1335.0889, 253.20589] 1, 3 vertical measurements, 2, 4 horizontal. its the location of 2 points that are connected 
        top_baseline = (court_detector.baseline_top[1] + court_detector.baseline_top[3]) / 2
        print(court_detector.baseline_bottom)
    # bottom_baseline
        bottom_baseline = (court_detector.baseline_bottom[2] + court_detector.baseline_bottom[2]) / 2
        # add players locations
        # img = mark_player_box(img, player1_boxes, frame_number)
        # img = mark_player_box(img, player2_boxes, frame_number)
        # img_no_frame = mark_player_box(img_no_frame, player1_boxes, frame_number)
        # img_no_frame = mark_player_box(img_no_frame, player2_boxes, frame_number)
MAYBE TRY AND PUT THE COURT CODE INTO OUR EXSISTING YOLO CODE, THAT WAY WE WILL AT LEAST HAVE THE COURT. THEN WE CAN USE THE BALL BOUNDING BOX LIKE THE PLAYER BOX 
        # add ball location
        img = ball_detector.mark_positions(img, frame_num=frame_number)
        img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')
        
        if (ball_detector.xy_coordinates[frame_number][0]) is not None:
            print(ball_detector.xy_coordinates[frame_number][0])
            print(ball_detector.xy_coordinates[frame_number][1])
            ball_y = ball_detector.xy_coordinates[frame_number][1]
            if (ball_y < top_baseline or ball_y > bottom_baseline):
                cv2.putText(img=img, text="OUT!!!", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,255), org=(100, 100))
                print("OUT!!!")
    # [676.44324, 255.8075, 1335.0889, 253.20589] 1, 3 vertical measurements, 2, 4 horizontal. its the location of 2 points that are connected 
        
        # print(court_detector.baseline_top[1])
        # print(court_detector.baseline_top[3])

 
        # add pose stickman
        # if skeleton_df is not None:
        #     img, img_no_frame = mark_skeleton(skeleton_df, img, img_no_frame, frame_number)

        # Add stroke prediction
        # for i in range(-10, 10):
        #     if frame_number + i in strokes_predictions.keys():
        #         '''cv2.putText(img, 'STROKE HIT', (200, 200),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255) if i != 0 else (255, 0, 0), 3)'''

        #         probs, stroke = strokes_predictions[frame_number + i]['probs'], strokes_predictions[frame_number + i][
        #             'stroke']
        #         cv2.putText(img, 'Forehand - {:.2f}, Backhand - {:.2f}, Service - {:.2f}'.format(*probs),
        #                     (70, 400),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #         cv2.putText(img, f'Stroke : {stroke}',
        #                     (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        #         break
        # Add stroke detected
        # for i in range(-5, 10):
        #     '''if frame_number + i in p1:
        #         cv2.putText(img, 'Stroke detected', (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)'''

        #     if frame_number + i in p2:
        #         cv2.putText(img, 'Stroke detected',
        #                     (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 50),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

        # cv2.putText(img, 'Distance: {:.2f} m'.format(player1_dists[frame_number] / 100),
        #             (50, 500),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        # cv2.putText(img, 'Distance: {:.2f} m'.format(player2_dists[frame_number] / 100),
        #             (100, 150),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # display frame
        if show_video:
            cv2.imshow('Output', img)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()

        # save output videos
        if with_frame == 0:
            final_frame = img_no_frame
        elif with_frame == 1:
            final_frame = img
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)
        out.write(final_frame)
        frame_number += 1

    print('Creating new video frames %d/%d  ' % (length, length), '\n', end='')
    print(f'New videos created, file name - {output_file}.avi')
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    s = time.time()
    detect_court()
    print(f'Total computation time : {time.time() - s} seconds')


if __name__ == "__main__":
    main()