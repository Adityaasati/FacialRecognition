import cv2
import mediapipe as mp


#Face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()



cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()

#Image
    # img = cv2.imread('person.jpg')
    # img = cv2.resize(img,(500,500))
    height, width, _ = img.shape
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



#Facial landmarks
    result = face_mesh.process(rgb_image)

    for facial_landmarks in result.multi_face_landmarks:
        for j in range(0,467):
            pt1 = facial_landmarks.landmark[j]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
 
            cv2.circle(img, (x,y), 2, (100,100,0), -1)
    
            # A = [x,y]
            # print(A)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

