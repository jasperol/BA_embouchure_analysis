import mediapipe
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize
from PIL import Image


mp_face_detection = mediapipe.solutions.face_detection
mp_face_mesh = mediapipe.solutions.face_mesh
face_detector = mp_face_detection.FaceDetection(min_detection_confidence = 0.6)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles



def crop_mark_aug(f_path, person_array, landmarks):

    face_list=[]
    
    for i, path in enumerate(person_array):
        
        image_path = os.path.join(f_path, path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height,img_width,_ = img.shape
        
        lm_result = face_mesh.process(img)
            
        
        
        if landmarks == True and lm_result.multi_face_landmarks != None:
            
            for facial_landmarks in lm_result.multi_face_landmarks:
                
                for i in range(0,468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * img_width)
                    y = int(pt1.y * img_height)
        
                    cv2.circle(img, (x,y), 2, (100,100,0), -1)
    
            
        results = face_detector.process(img)
            
        if results.detections != None:
            face = results.detections[0]
            bounding_box = face.location_data.relative_bounding_box

            # Calculate absolute bounding box coordinates
            abs_xmin = int(bounding_box.xmin * img_width)
            abs_ymin = int(bounding_box.ymin * img_height)
            abs_width = int(bounding_box.width * img_width)
            abs_height = int(bounding_box.height * img_height)
    # Crop the face from the original image
            face_image = img[abs_ymin + round(1/2 * abs_height):abs_ymin + abs_height, abs_xmin:abs_xmin + abs_width]
            #face_image = img[abs_ymin + round(abs_height*1/2):abs_ymin + abs_height, abs_xmin + round(abs_width*1/6):abs_xmin + round(abs_width*5/6)]

            face_list.append(face_image)
            
    filtered_images = [img for img in face_list if img.shape[0]>50 and img.shape[1]>50]

    re_face_list =[]
    for i, face in enumerate(filtered_images):
        re_face = resize(face, (224,224))
        re_face_list.append(re_face)

    array_face_list = np.stack(re_face_list)
    
    aug_face_list = []

    for i, face in enumerate(array_face_list[::2]):
        flip_face = cv2.flip(face,1)
        noise = np.random.normal(0, 0.05, face.shape)
        noisy_image = np.clip(face + noise, 0, 1)
        angle = np.random.uniform(-15, 15)
        height, width = face.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_image = cv2.warpAffine(face, rotation_matrix, (width, height))
        aug_face_list.extend([flip_face, face])
        
    aug_array_face_list = np.stack(aug_face_list)

    return array_face_list, aug_array_face_list



vid = "_"
folder_path = f"frames/{vid}"
file_list = os.listdir(folder_path)
name = "_"

array = [file for file in file_list if name in file]
array_1, aug_array1 = crop_mark_aug(folder_path,array, True)
array_2, aug_array2 = crop_mark_aug(folder_path,array, False)

np.save(f"prepro_data/{name}_{vid}_marks.npy", aug_array1)
np.save(f"prepro_data/{name}_{vid}.npy", aug_array2)