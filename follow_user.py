#import libraries
import cv2
import face_recognition

#capture webcam
video=cv2.VideoCapture('/dev/video0')

#Load User Image to identify him
image=face_recognition.load_image_file('PrashnatAnuragi_PassportPic.jpg')
face_encoding=face_recognition.face_encodings(image)[0]

known_faces=[face_encoding,]

#initialize variables
face_locations=[]
face_encodings=[]
face_names=[]
frame_number=0

while True:
    ret,frame=video.read()

    rgb_frame=frame[:,:,::-1]

    #find the faces
    #face_locations=face_recognition.face_locations(rgb_frame,model='cnn')
    face_locations=face_recognition.face_locations(rgb_frame)
    face_encodings=face_recognition.face_encodings(rgb_frame,face_locations)

    face_names=[]
    for face_encoding in face_encodings:
        #if face matches with known face
        match=face_recognition.compare_faces(known_faces,face_encoding,tolerance=0.50)

        name=None
        if match[0]:
            name='Prashant'

        face_names.append(name)

    #Label the results
    for (top,right,bottom,left), name in zip(face_locations,face_names):
        if not name:
            continue

        #Draw a box around the face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)

        #Draw a label with a name below the face
        cv2.rectangle(frame,(left,bottom-25),(right,bottom),(0,255,0),cv2.FILLED)

        font=cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame,name,(left+6,bottom-6),font,0.5,(255,255,255),1)

        #Display the resultant image
        cv2.imshow('Video',frame)

    #Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()


