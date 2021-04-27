#import libraries
import cv2
import face_recognition

#Get a reference to webcam
#to find the device run 'ls /dev/video*' on your terminal
video=cv2.VideoCapture('/dev/video0')

#initialize variables
face_locations_=[]
while True:
    #grab a single fram of video
    ret,frame=video.read()

    #Convert the image from BGR (opencv uses this) to RGB (face_recognition uses this)
    rgb_frame=frame[:,:,::-1]

    #find the faces in the current frame of video
    face_locations_=face_recognition.face_locations(rgb_frame)

    #Display the results
    for top,right,bottom,left in face_locations_:
        #Draw a box around the face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)

    #Display the resultant image
    cv2.imshow('Video',frame)

    #Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Release webcam handle
video.release()
cv2.destroyAllWindows()

