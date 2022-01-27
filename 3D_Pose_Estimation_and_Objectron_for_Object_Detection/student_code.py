import os
import numpy as np
import cv2
import mediapipe as mp
from proj6_code.utils import *

# TO-DO 1
def detect_3d_box(img_path):

    '''
        Given an image, this function detects the 3D bounding boxes' 8 vertices of the chair in the image.
        We will only consider one chair in one single image.
        Similar to pose estimation, you're going to use mediapipe to detect the 3D bounding boxes.
        You should try to understand how does the objectron work before trying to finish this function!

        Args:
        -    img_path: the path of the RGB chair image

        Returns:
        -

        boxes: numpy array of 2D points, which represents the 8 vertices of 3D bounding boxes
        annotated_image: the original image with the overlapped bounding boxes

        Useful functions for usage: inference()
    '''

    model_path = 'object_detection_3d_chair.tflite'
    if os.path.exists('../object_detection_3d_chair.tflite'):
        model_path = "../object_detection_3d_chair.tflite"
    elif os.path.exists('../../object_detection_3d_chair.tflite'):
        model_path = "../../object_detection_3d_chair.tflite"

    boxes = None
    hm = None
    displacements = None

    inshapes = [[1, 3, 640, 480]]
    outshapes = [[1, 16, 40, 30], [1, 1, 40, 30]]
    print(inshapes, outshapes)

    if img_path == 'cam':
        cap = cv2.VideoCapture(0)

    while True:
        if img_path == 'cam':
            _, img_orig = cap.read()
        else:
            img_file = img_path
            img_orig = cv2.imread(img_file)

        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (inshapes[0][3], inshapes[0][2]))
        img = img.transpose((2,0,1))
        image = np.array(img, np.float32)/255.0
        ############################################################################
        # TODO: YOUR CODE HERE
        ############################################################################
        # Step-1: Call inference and get the result.
        hm, displacements = inference(image, model_path)
        
        # Step-2: Decode bounding boxes from inference result . 
        boxes = decode(hm, displacements)


        
        
        ############################################################################
        #                             END OF YOUR CODE
        ############################################################################

        # Draw the bounding box.
        for obj in boxes:
            draw_box(img_orig, obj)
        return boxes[0], cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)


# TO-DO 2
def hand_pose_img(test_img):
    """
        Given an image, it calculates the pose of human in the image.
        To make things easier, we only consider one people on a single image.
        Pose estimation is actually a difficult problem, in this function, you are going to use
        mediapipe to do this work. You can find more about mediapipe from its official website
        https://google.github.io/mediapipe/solutions/pose#overview

        Args:
        -    img: path to rgb image

        Returns:
        -    landmark: numpy array of size (n, 2) the landmark detected by mediapipe,
        where n is the length of landmark, 2 represents x and y coordinates
        (Note, not in the range 0-1, you need to get the real 2D coordinates in images)

        the order of these landmark should be consistent with the original order returned by mediapipe
        -    annotated_image: the original image overlapped with the detected landmark

        Useful functions/class: mediapipe.solutions.pose, mediapipe.solutions.drawing_utils
    """

    landmark = None
    annotated_image = None

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Init pose model and read the image.
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(test_img)

    # Convert the BGR image to RGB before processing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows = image.shape[0]
    cols = image.shape[1]

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    # Reminder: check the mediapipe documentation first!
    # Step-1: Pass the image to the pose model.
    results = pose.process(image)
    
    # Step-2: Get landmark from the result.
    landmark = results.pose_landmarks
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    # Display the pose.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
      annotated_image, landmark, mp_pose.POSE_CONNECTIONS)
    pose.close()
    landmark1=np.zeros((len(results.pose_landmarks.landmark),2))
    for i in range(len(results.pose_landmarks.landmark)):
        landmark1[i,:] = [results.pose_landmarks.landmark[i].x*cols, results.pose_landmarks.landmark[i].y*rows]
    return landmark1, annotated_image


# TO-DO 3
def check_hand_inside_bounding_box(hand, pts):
    """
    This function checks whether the hand is inside the bounding box of the
    chair or not.
    Args:
        hand: 3D coordinate of the hand (numpy.array, size 1*3)
        pts: 3D coordinates of the 8 vertices of the bounding box (numpy.array, size 8*3)
    Returns:
        inside: boolean value, True if hand is inside the bounding box, and
                False otherwise.

    Hint: Build a coordinate system along the edges of the bounding box, map the pts and
    hand points to the new coordinate system. This will simplify the comparison process
    especially when the bounding box edges are not perfectly vertical or horizontal.

    To do coordinate transform, the suggested method is to use np.amin() to get the minimum
    value for each dimension from the pts, then subtract the result of np.amin() from original
    hand and pts coordinate. In this way, the point returned by np.amin() becomes the (0, 0, 0)
    point in the new coordinate system. Then check if each dimension of the transfromed hand
    coordinate is within np.amax() of the transformed pts and (0, 0, 0).
    """

    inside = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    transform = np.amin(pts[:,:3], axis = 0)
    boundary = np.amax(pts[:,:3] - [transform],axis = 0)
    hand_transformed = hand - transform
    
    inside = np.all(np.less_equal(hand_transformed, boundary)) and np.all(np.less_equal([0,0,0],hand_transformed))
    
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return inside


# TO-DO EC Video
def process_video(path):
    """
    This function will process the video that you take and should output a video
    that shows you interacting with one or two chairs and their bounding boxes changing colors.

    Args:
        path: a path to the your video file

    Returns:
        none (But a video file should be generated)

    The recommended approach is to process your video mp4 using cv2.VideoCapture.
    For usage you can look up the official opencv documentation.
    You can split up your video into individual frames, and process each frame
    like we did in the notebook, with the correct parameters and correct calibration.
    These individual frames can be turned back into a video, which you can save to your
    computer.

    A simple tutorial can be found here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    """

    count = 0
    vidcap = cv2.VideoCapture(path)
    def processFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            img = image
            img_path = "../data/preprocessed/image"+str(count)+".jpg"
            if os.path.exists(img_path):
                return hasFrames
            cv2.imwrite(img_path, img)

            # Do calibration.
            width = 0.91
            height = 0.91
            depth = 1.3
            vertices_world = get_world_vertices(width, height, depth)
            initial_box_points_3d = vertices_world
            path = '../data/cali/example/' # update the path to where you save the pictures
            path2 = '../data/cali2/example2/' # update the path to where you save the pictures
            path3 = '../data/cali3/'

            K = calibrate(path3) # intrinsic matrix

            try:
                bounding_boxes_chair_2d, a_img = detect_3d_box(img_path)
            except NotImplementedError:
                print("Implement detect_3d_box first.")

            bounding_boxes = bounding_boxes_chair_2d
            height = a_img.shape[0]
            width = a_img.shape[1]

            box_points_2d = np.array(bounding_boxes)
            box_points_2d[:, 0] *= width
            box_points_2d[:, 1] *= height

            projection_depth = 2.5

            ############################################################################
            # TODO: YOUR CODE HERE
            ############################################################################

            # Step-1: Get the projection matrix by calling perspective_n_points()
            
            # Step-2: Call hand_pose_img to get landmarks from img_path
            
            # Step-3: Project 2d landmakrs to 3d using given projection_depth and project matrix from step-3
            
            # Step-4: Get hand coordinate from the index 22 (0-indexed) of 3d landmarks
            
            # Step-5 Call draw_box_intersection and get annotated_img
            
            ############################################################################
            #                             END OF YOUR CODE
            ############################################################################

            cv2.imwrite("../data/video/image"+str(count)+".jpg", annotated_img)     # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 1/30 # 30 fps
    count=1
    success = processFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        #print(sec)
        success = processFrame(sec)

    pathIn= '../data/video/'
    pathOut = '../data/extra_credit_video/extra_credit_video.mp4'

    fps = 30
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
    files = list(filter(lambda x: x[-4:] == ".jpg", files))
    files.sort(key = lambda x: int(x[5:-4]))

    size = 0
    for i in range(len(files)):
        filename = pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)

        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    print('Video outputted to: {}'.format(pathOut))


def draw_box_intersection(image, hand, pts, pts_2d):
    """
    Draw the bounding box (in blue) around the chair. If the hand is within the
    bounding box, then we draw it with another color (red)
    Args:
        image: the image in which we'll draw the bounding box, the channel follows RGB order
        hand: 3D coordinate of the hand (numpy.array, 1*3)
        pts: 3D coordinates of the 8 vertices of the bounding box (numpy.array, 8*3)
        pts_2d: 2D coordinates of the 8 vertices of the bounding box (numpy.array, 8*2)

    Returns:
        image: annotated image
    """
    if np.shape(pts)[1] == 3:
        pts = np.concatenate([pts, np.ones((8,1))], axis=1)

    color = (0, 0, 0)
    if check_hand_inside_bounding_box(hand, pts):
        color = (0, 255, 255)
        print("Check succeed!")

    thickness = 5

    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0, 1), (1, 3), (0, 2), (3, 2), (1, 5), (0, 4), (2, 6), (3, 7), (5, 7), (6, 7), (6, 4), (4, 5)]
    for line in lines:
        pt0 = pts_2d[line[0]]
        pt1 = pts_2d[line[1]]
        pt0 = (int(pt0[0] * scaleX), int(pt0[1] * scaleY))
        pt1 = (int(pt1[0] * scaleX), int(pt1[1] * scaleY))
        cv2.line(image, pt0, pt1, color, thickness)


    for i in range(8):
        pt = pts_2d[i]
        pt = (int(pt[0] * scaleX), int(pt[1] * scaleY))
        cv2.circle(image, pt, 8, (0, 255, 0), -1)
        cv2.putText(image, str(i), pt, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    print(image.shape)
    return image
