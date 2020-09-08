import cv2
import numpy as np
import dlib
import imutils.face_utils

def get_pose_data(size, face):
    #print(type(face))
    shape0 = np.array(face)


    image_points = np.array([
                                (shape0[30, :]),     # Nose tip
                                (shape0[8,  :]),     # Chin
                                (shape0[36, :]),     # Left eye left corner
                                (shape0[45, :]),     # Right eye right corne
                                (shape0[48, :]),     # Left Mouth corner
                                (shape0[54, :])      # Right mouth corner
                            ], dtype="double")

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner                     
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    #print ("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
    #print ("Rotation Vector:\n {0}".format(rotation_vector))
    #print ("Translation Vector:\n {0}".format(translation_vector))
    return {"rotation_vector":rotation_vector,"translation_vector":translation_vector,"camera_matrix":camera_matrix,"dist_coeffs":dist_coeffs, "image_points":image_points}

def categorize_look(stats):
    lr = stats["rotation_vector"][2][0]
    #print(lr)
    ud = stats["rotation_vector"][1]
    lr_boundry = 0.6
    lr_cat = "centre"
    if(lr > lr_boundry):
        lr_cat = "right"
    elif(lr < (0- lr_boundry)):
        lr_cat = "left"
    return lr_cat
def get_pose(size,face):
    stats = get_pose_data(size,face)
    pose = categorize_look(stats)
    return pose