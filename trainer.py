#YOLOv7 Bicep Count
#Internal use at MILKI-PSY project

import math
import cv2

# =3.0=Takes three points and returns the angle between them========
def findAngle(image, kpts, p1,p2,p3, draw= True):
    coord = []
    no_kpt = len(kpts)//3
    for i in range(no_kpt):
        cx,cy = kpts[3*i], kpts[3*i +1]
        conf = kpts[3*i +2]
        coord.append([i, cx,cy, conf])

    points = (p1,p2,p3)

    # =3.1=Get landmarks========
    x1,y1 = coord[p1][1:3]
    x2,y2 = coord[p2][1:3]
    x3,y3 = coord[p3][1:3]

    # =3.2=Calculate the Angle========
    angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle += 360

    # =3.3=Draw Skeleton with Circles========
    if draw:
        # Draw white line between keypoints
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
        cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), (255, 0, 0), 10)

        # Draw white circles at keypoints
        cv2.circle(image, (int(x1), int(y1)), 20, (255, 0, 0), cv2.FILLED)
        cv2.circle(image, (int(x2), int(y2)), 20, (255, 0, 0), cv2.FILLED)
        cv2.circle(image, (int(x3), int(y3)), 20, (255, 0, 0), cv2.FILLED)
    return int(angle)

    # =3.3=Draw Coordinates========
    #if draw:
        #cv2.line(image, (int(x1),int(y1)),(int(x2), int(y2)),(255,0,255),3 )
        #cv2.line(image, (int(x3),int(y3)),(int(x2), int(y2)),(255,0,255),3 )

        #cv2.circle(image, (int(x1),int(y1)),  10, (255,255,255),cv2.FILLED)
        #cv2.circle(image, (int(x1), int(y1)), 20, (235, 235, 235), 5)
        #cv2.circle(image, (int(x1), int(y1)), 10, (255, 255, 255), cv2.FILLED)
        #cv2.circle(image, (int(x1), int(y1)), 20, (235, 235, 235), 5)
        #cv2.circle(image, (int(x1), int(y1)), 10, (255, 255, 255), cv2.FILLED)
        #cv2.circle(image, (int(x1), int(y1)), 20, (235, 235, 235), 5)

    #return int(angle)
# ===========================================