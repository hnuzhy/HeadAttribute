
from PyQt4.QtGui import *
from PyQt4.QtCore import *

import numpy as np
from math import cos, sin

def calculate_cube(bbox_list, yaw, roll, pitch, plot_prismatic=False):
    
    [face_x, face_y, face_w, face_h] = bbox_list
    size = int((face_w + face_h)/2)

    '''The changing is only for visualization'''
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    p = pitch * np.pi / 180

        
    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    
    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    
    # Z-Axis (out of the screen) drawn in blue
    # x3 = size * (sin(y)) + face_x
    # y3 = size * (-cos(y) * sin(p)) + face_y
    # or Z-Axis (inside the screen) drawn in blue
    x3 = face_x - size * (sin(y))
    y3 = face_y - size * (-cos(y) * sin(p))


    # Draw base in red
    # cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)), (0,0,255), 3)
    # cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)), (0,0,255), 3)
    # cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)), (0,0,255), 3)
    # cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)), (0,0,255), 3)
    pts_red1 = [round(face_x, 2), round(face_y, 2)]
    pts_red2 = [round(x1, 2), round(y1, 2)]
    pts_red3 = [round(x2+x1-face_x, 2), round(y2+y1-face_y, 2)]
    pts_red4 = [round(x2, 2), round(y2, 2)]
    
    # Draw pillars in blue
    # cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    # cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    # cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    # cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    
    # Draw top in green
    # cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    # cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    # cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    # cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)
    pts_green1 = [round(x3, 2), round(y3, 2)]
    pts_green2 = [round(x3+x1-face_x, 2),round(y3+y1-face_y, 2)]
    pts_green3 = [round(x3+x1+x2-2*face_x, 2),round(y3+y2+y1-2*face_y, 2)]
    pts_green4 = [round(x3+x2-face_x, 2),round(y3+y2-face_y, 2)]


    fourpts = [pts_red1, pts_red2, pts_red4, pts_green1]  # for three axises plotting
    
    ''' Plot prismatic instead of cube for better visualization '''
    lambda_split = 0.2
    if plot_prismatic:
        temp_len_x = (pts_red3[0] - pts_red1[0]) * lambda_split
        temp_len_y = (pts_red3[1] - pts_red1[1]) * lambda_split
        pts_red1[0], pts_red1[1] = pts_red1[0] + temp_len_x, pts_red1[1] + temp_len_y
        pts_red3[0], pts_red3[1] = pts_red3[0] - temp_len_x, pts_red3[1] - temp_len_y

        temp_len_x = (pts_red4[0] - pts_red2[0]) * lambda_split
        temp_len_y = (pts_red4[1] - pts_red2[1]) * lambda_split
        pts_red2[0], pts_red2[1] = pts_red2[0] + temp_len_x, pts_red2[1] + temp_len_y
        pts_red4[0], pts_red4[1] = pts_red4[0] - temp_len_x, pts_red4[1] - temp_len_y
    
    ''' Align eight points of cube with the face center point (face_x, face_y) '''
    eightpts = [pts_red1, pts_red2, pts_red3, pts_red4, pts_green1, pts_green2, pts_green3, pts_green4]
    max_x = min_x = face_x
    max_y = min_y = face_y
    for point in eightpts:
        max_x, min_x = max(point[0], max_x), min(point[0], min_x)
        max_y, min_y = max(point[1], max_y), min(point[1], min_y)
    cube_center_x = (min_x + max_x) / 2.0
    cube_center_y = (min_y + max_y) / 2.0
    
    x_offset = cube_center_x - face_x
    y_offset = cube_center_y - face_y
    
    aligned_eightpts = []
    for point in eightpts:
        aligned_eightpts.append([point[0] - x_offset, point[1] - y_offset])

    return aligned_eightpts, fourpts
    

def calculate_octahedron(eightpts, nose_ptx=0, nose_pty=0):
    # https://en.wikipedia.org/wiki/Platonic_solid
    # Every polyhedron has a dual (or "polar") polyhedron with faces and vertices interchanged.
    # The cube (regular hexahedron) and the octahedron form a dual pair.
    # Visual Polyhedra: http://dmccooey.com/polyhedra/index.html
    
    center_x, center_y = 0, 0
    for point in eightpts:
        center_x += point[0]
        center_y += point[1]
    center_x = int(center_x / len(eightpts))
    center_y = int(center_y / len(eightpts)) 
    
    pts_index_dict = {
        "front": [0,1,2,3],     # index 0
        "back": [4,5,6,7],      # index 1
        "left": [0,3,4,7],      # index 2 
        "right": [1,2,5,6],     # index 3
        "top": [0,1,4,5],       # index 4
        "bottom": [2,3,6,7]     # index 5
        }
        
    sixpts = []
    for pts_name, index_list in pts_index_dict.items():
        pt_x, pt_y = 0, 0
        for index in index_list:
            pt_x += eightpts[index][0]
            pt_y += eightpts[index][1]
        sixpts.append([pt_x/4, pt_y/4])
     
    # Align the calculated dual octahedron with face nose landmark ( == "front" point)
    # Or directly refer https://github.com/anilbas/BFMLandmarks       
    ptx_offset = sixpts[0][0] - sixpts[0][0] - nose_ptx 
    pty_offset = sixpts[0][1] - sixpts[0][1] - nose_pty
    
    sixpts_aligned = []
    for point in sixpts:
        sixpts_aligned.append([point[0] - ptx_offset, point[1] - pty_offset])

    return sixpts_aligned, [center_x, center_y]
    

def plot_3d_axises_etc(painter, eightpts, fourpts, scale, yaw, visualization_flag="cube_mayavi"):
    # "three_axises" or "triangular_pyramid" or "cube_positive" or "cube_negative" or "cube_mayavi"

    penRed = QPen(QColor(255,0,0), max(1, int(round(2.0 / scale))), Qt.SolidLine)
    penGreen = QPen(QColor(0,255,0), max(1, int(round(2.0 / scale))), Qt.SolidLine)
    penBlue = QPen(QColor(0,0,255), max(1, int(round(2.0 / scale))), Qt.SolidLine)
    penYellow = QPen(QColor(255,255,0), max(1, int(round(2.0 / scale))), Qt.SolidLine)
    penBlack = QPen(QColor(0,0,0), max(2, int(round(4.0 / scale))), Qt.SolidLine)
    penGray = QPen(QColor(127,127,127), max(2, int(round(4.0 / scale))), Qt.SolidLine)

    
    if visualization_flag == "three_axises":
        painter.setPen(penRed)  # red  
        painter.drawLine(fourpts[0][0], fourpts[0][1], fourpts[1][0], fourpts[1][1])   
        painter.setPen(penGreen)  # green 
        painter.drawLine(fourpts[0][0], fourpts[0][1], fourpts[2][0], fourpts[2][1])
        painter.setPen(penBlue)  # blue  
        painter.drawLine(fourpts[0][0], fourpts[0][1], fourpts[3][0], fourpts[3][1])
        painter.setPen(penBlack)
        painter.drawPoint(fourpts[0][0], fourpts[0][1])
        
    if visualization_flag == "triangular_pyramid":
        painter.setPen(penRed)  # red  
        painter.drawLine(fourpts[0][0], fourpts[0][1], fourpts[1][0], fourpts[1][1])   
        painter.setPen(penGreen)  # green 
        painter.drawLine(fourpts[0][0], fourpts[0][1], fourpts[2][0], fourpts[2][1])
        painter.setPen(penBlue)  # blue  
        painter.drawLine(fourpts[0][0], fourpts[0][1], fourpts[3][0], fourpts[3][1])
        
        painter.setPen(penBlack)
        painter.drawPoint(fourpts[0][0], fourpts[0][1])
        
        painter.setPen(penYellow)  # blue
        triangular_pts = [QPoint(fourpts[1][0], fourpts[1][1]), QPoint(fourpts[2][0], fourpts[2][1]), 
            QPoint(fourpts[3][0], fourpts[3][1]), QPoint(fourpts[1][0], fourpts[1][1])]
        frontal_path = QPainterPath()
        frontal_path.addPolygon(QPolygonF(triangular_pts))
        painter.drawPath(frontal_path)
        painter.fillPath(frontal_path, QColor(127, 127, 127, 150))  # RGBA
        
    if visualization_flag == "cube_positive":
        
        painter.setPen(penRed)  # red
        base_pts = [QPoint(eightpts[0][0], eightpts[0][1]), QPoint(eightpts[1][0], eightpts[1][1]),
            QPoint(eightpts[2][0], eightpts[2][1]), QPoint(eightpts[3][0], eightpts[3][1])]
        painter.drawPolygon(QPolygon(base_pts))
        
        painter.setPen(penGreen)  # green
        for i in range(4):
            start_x , start_y = eightpts[i][0], eightpts[i][1]
            end_x, end_y = eightpts[i+4][0], eightpts[i+4][1]
            painter.drawLine(start_x , start_y, end_x, end_y)
        
        painter.setPen(penBlue)  # blue
        front_pts = [QPoint(eightpts[4][0], eightpts[4][1]), QPoint(eightpts[5][0], eightpts[5][1]),
            QPoint(eightpts[6][0], eightpts[6][1]), QPoint(eightpts[7][0], eightpts[7][1])]
        frontal_path = QPainterPath()
        frontal_path.addPolygon(QPolygonF(front_pts+[QPoint(eightpts[4][0], eightpts[4][1])]))
        painter.drawPath(frontal_path)
        painter.fillPath(frontal_path, QColor(127, 127, 127, 150))  # RGBA
        
        painter.setPen(penYellow)  # yellow
        painter.drawLine(eightpts[0][0], eightpts[0][1], eightpts[5][0], eightpts[5][1])
        painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[4][0], eightpts[4][1])
        
        painter.setPen(penBlack)  # black
        painter.drawPoint(eightpts[0][0], eightpts[0][1])
        painter.setPen(penGray)  # gray
        painter.drawPoint(eightpts[1][0], eightpts[1][1])
        
    if visualization_flag == "cube_negative":
        if abs(yaw) < 90:
            painter.setPen(penRed)  # red
            base_pts = [QPoint(eightpts[4][0], eightpts[4][1]), QPoint(eightpts[5][0], eightpts[5][1]),
                QPoint(eightpts[6][0], eightpts[6][1]), QPoint(eightpts[7][0], eightpts[7][1])]
            painter.drawPolygon(QPolygon(base_pts))
            
            painter.setPen(penGreen)  # green
            for i in range(4):
                start_x , start_y = eightpts[i][0], eightpts[i][1]
                end_x, end_y = eightpts[i+4][0], eightpts[i+4][1]
                painter.drawLine(start_x , start_y, end_x, end_y)
            
            painter.setPen(penBlue)  # blue
            front_pts = [QPoint(eightpts[0][0], eightpts[0][1]), QPoint(eightpts[1][0], eightpts[1][1]),
                QPoint(eightpts[2][0], eightpts[2][1]), QPoint(eightpts[3][0], eightpts[3][1])]
            frontal_path = QPainterPath()
            frontal_path.addPolygon(QPolygonF(front_pts+[QPoint(eightpts[0][0], eightpts[0][1])]))
            painter.drawPath(frontal_path)
            painter.fillPath(frontal_path, QColor(127, 127, 127, 150))  # RGBA
            
        else:
            painter.setPen(penBlue)  # blue
            front_pts = [QPoint(eightpts[0][0], eightpts[0][1]), QPoint(eightpts[1][0], eightpts[1][1]),
                QPoint(eightpts[2][0], eightpts[2][1]), QPoint(eightpts[3][0], eightpts[3][1])]
            frontal_path = QPainterPath()
            frontal_path.addPolygon(QPolygonF(front_pts+[QPoint(eightpts[0][0], eightpts[0][1])]))
            painter.drawPath(frontal_path)
            painter.fillPath(frontal_path, QColor(127, 127, 127, 150))  # RGBA

            painter.setPen(penGreen)  # green
            for i in range(4):
                start_x , start_y = eightpts[i][0], eightpts[i][1]
                end_x, end_y = eightpts[i+4][0], eightpts[i+4][1]
                painter.drawLine(start_x , start_y, end_x, end_y)

            painter.setPen(penRed)  # red
            base_pts = [QPoint(eightpts[4][0], eightpts[4][1]), QPoint(eightpts[5][0], eightpts[5][1]),
                QPoint(eightpts[6][0], eightpts[6][1]), QPoint(eightpts[7][0], eightpts[7][1])]
            painter.drawPolygon(QPolygon(base_pts))

        painter.setPen(penYellow)  # yellow
        painter.drawLine(eightpts[0][0], eightpts[0][1], eightpts[5][0], eightpts[5][1])
        painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[4][0], eightpts[4][1])

        painter.setPen(penBlack)  # black
        painter.drawPoint(eightpts[0][0], eightpts[0][1])
        painter.setPen(penGray)  # gray
        painter.drawPoint(eightpts[1][0], eightpts[1][1])
        
    if visualization_flag == "cube_mayavi":
    
        center_x, center_y = 0, 0
        for point in eightpts:
            center_x += point[0]
            center_y += point[1]
        center_x = int(center_x / len(eightpts))
        center_y = int(center_y / len(eightpts)) 
        center_pt = [center_x, center_y]
        
        painter.setPen(penRed)  # red  
        painter.drawLine(center_pt[0], center_pt[1], (eightpts[1][0]+eightpts[6][0])/2, (eightpts[1][1]+eightpts[6][1])/2)
        painter.setPen(penGreen)  # green 
        painter.drawLine(center_pt[0], center_pt[1], (eightpts[1][0]+eightpts[4][0])/2, (eightpts[1][1]+eightpts[4][1])/2)
        painter.setPen(penBlue)  # blue 
        painter.drawLine(center_pt[0], center_pt[1], (eightpts[1][0]+eightpts[3][0])/2, (eightpts[1][1]+eightpts[3][1])/2)       
        painter.setPen(penYellow)  # yellow
        painter.drawPoint(center_pt[0], center_pt[1])

        if abs(yaw) < 90:
            painter.setPen(penRed)  # red  
            painter.drawLine(eightpts[7][0], eightpts[7][1], eightpts[6][0], eightpts[6][1])  
            painter.setPen(penGreen)  # green 
            painter.drawLine(eightpts[7][0], eightpts[7][1], eightpts[4][0], eightpts[4][1])
            painter.setPen(penBlue)  # blue  
            painter.drawLine(eightpts[7][0], eightpts[7][1], eightpts[3][0], eightpts[3][1])
        
            penBlack = QPen(QColor(0,0,0), max(1, int(round(2.0 / scale))), Qt.SolidLine)
            painter.setPen(penBlack)  # black
            pts = [QPoint(eightpts[6][0], eightpts[6][1]), QPoint(eightpts[5][0], eightpts[5][1]),
                QPoint(eightpts[4][0], eightpts[4][1]), QPoint(eightpts[0][0], eightpts[0][1]),
                QPoint(eightpts[3][0], eightpts[3][1]), QPoint(eightpts[2][0], eightpts[2][1]),
                QPoint(eightpts[6][0], eightpts[6][1])]
            q_path = QPainterPath()
            q_path.addPolygon(QPolygonF(pts))
            painter.drawPath(q_path)
            
            painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[0][0], eightpts[0][1])
            painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[2][0], eightpts[2][1])
            painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[5][0], eightpts[5][1])
        else:
            penBlack = QPen(QColor(0,0,0), max(1, int(round(2.0 / scale))), Qt.SolidLine)
            painter.setPen(penBlack)  # black
            pts = [QPoint(eightpts[6][0], eightpts[6][1]), QPoint(eightpts[5][0], eightpts[5][1]),
                QPoint(eightpts[4][0], eightpts[4][1]), QPoint(eightpts[0][0], eightpts[0][1]),
                QPoint(eightpts[3][0], eightpts[3][1]), QPoint(eightpts[2][0], eightpts[2][1]),
                QPoint(eightpts[6][0], eightpts[6][1])]
            q_path = QPainterPath()
            q_path.addPolygon(QPolygonF(pts))
            painter.drawPath(q_path)
            
            painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[0][0], eightpts[0][1])
            painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[2][0], eightpts[2][1])
            painter.drawLine(eightpts[1][0], eightpts[1][1], eightpts[5][0], eightpts[5][1])
        
            painter.setPen(penRed)  # red  
            painter.drawLine(eightpts[7][0], eightpts[7][1], eightpts[6][0], eightpts[6][1])  
            painter.setPen(penGreen)  # green 
            painter.drawLine(eightpts[7][0], eightpts[7][1], eightpts[4][0], eightpts[4][1])
            painter.setPen(penBlue)  # blue  
            painter.drawLine(eightpts[7][0], eightpts[7][1], eightpts[3][0], eightpts[3][1])
            
    if visualization_flag == "octahedron":
        sixpts, center_pt = calculate_octahedron(eightpts)
        
        painter.setPen(penRed)  # red 
        pts = [QPoint(sixpts[0][0], sixpts[0][1]), QPoint(sixpts[4][0], sixpts[4][1]),
            QPoint(sixpts[1][0], sixpts[1][1]), QPoint(sixpts[5][0], sixpts[5][1]),
            QPoint(sixpts[0][0], sixpts[0][1])]
        q_path = QPainterPath()
        q_path.addPolygon(QPolygonF(pts))
        painter.drawPath(q_path)
        
        painter.setPen(penGreen)  # green 
        pts = [QPoint(sixpts[0][0], sixpts[0][1]), QPoint(sixpts[2][0], sixpts[2][1]),
            QPoint(sixpts[1][0], sixpts[1][1]), QPoint(sixpts[3][0], sixpts[3][1]),
            QPoint(sixpts[0][0], sixpts[0][1])]
        q_path = QPainterPath()
        q_path.addPolygon(QPolygonF(pts))
        painter.drawPath(q_path)

        painter.setPen(penBlue)  # blue 
        pts = [QPoint(sixpts[2][0], sixpts[2][1]), QPoint(sixpts[4][0], sixpts[4][1]),
            QPoint(sixpts[3][0], sixpts[3][1]), QPoint(sixpts[5][0], sixpts[5][1]),
            QPoint(sixpts[2][0], sixpts[2][1])]
        q_path = QPainterPath()
        q_path.addPolygon(QPolygonF(pts))
        painter.drawPath(q_path)

        painter.setPen(penRed)  # red  
        painter.drawLine(center_pt[0], center_pt[1], sixpts[3][0], sixpts[3][1])
        painter.setPen(penGreen)  # green 
        painter.drawLine(center_pt[0], center_pt[1], sixpts[4][0], sixpts[4][1])
        painter.setPen(penBlue)  # blue 
        painter.drawLine(center_pt[0], center_pt[1], sixpts[0][0], sixpts[0][1])        
        painter.setPen(penBlack)  # black
        painter.drawPoint(center_pt[0], center_pt[1])