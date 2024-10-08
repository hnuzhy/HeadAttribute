#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt4.QtGui import *
from PyQt4.QtCore import *
    
from traitsui.api import View, Item
from traits.api import HasTraits, Instance, on_trait_change

from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor


import math
import numpy as np

from libs.mlab_3D_to_2D import get_world_to_view_matrix 
from libs.mlab_3D_to_2D import get_view_to_display_matrix 
from libs.mlab_3D_to_2D import apply_transform_to_points 

# head_model_path = "D:/dataset/2021-10-16_PersonHeadPose/Head_Attributes_v4/data/Female3DHead.obj"
head_model_path = "./data/Female3DHead.obj"

################################################################################
# The actual visualization
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    # This function is called when the view is opened. 
    # We don't populate the scene when the view is not yet open, 
    # as some VTK features require a GLContext. 
    # We can do normal mlab calls on the embedded scene.
    @on_trait_change('scene.activated')
    def update_plot(self):

        # self.scene.mlab.test_points3d()
        # self.scene.mlab.pipeline.surface(self.scene.mlab.pipeline.open("cylinder.vtk"))
        self.scene.mlab.pipeline.surface(self.scene.mlab.pipeline.open(head_model_path))
        
        pass
        
    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
        height=200, width=120, show_label=False, resizable=True), resizable=True)


################################################################################
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt input hook.
        # QtCore.pyqtRemoveInputHook()  # in QtCore
        # import pdb
        # pdb.set_trace()
        # QtCore.pyqtRestoreInputHook()  # in QtCore

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)
        
        
        # https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html
        
        self.visualization.scene.mlab.view(0, 0, roll=0)  # init
        self.init_view = self.visualization.scene.mlab.view()
        
        # Mlab.view(azimuth=None,elevation=None,distance=None,focalpoint=None,roll=None,Reset_roll=True,figure=None)
        print("view: ", self.visualization.scene.mlab.view())
        
        
        # print(self.visualization.scene)  # MlabSceneModel Instance
        # print(self.visualization.scene.get_size())  # a [w,h] tuple
        # print(self.visualization.scene.camera.clipping_range)  # a [x1,x2] tuple
        # print(self.visualization.scene.camera.orientation)  # for camera view, not object view
        
        
        # self.four_pts = self.plot_3d_XYZ_lines()
        self.eight_pts = self.plot_3d_cube_ploygon()
        self.refer_lms = self.plot_3d_refer_landmarks()
        
        self.visualization.scene.parallel_projection = True  # for accurate 3d points to 2d projection
        # self.visualization.scene.axes_indicator = True  # for better 3d pose checking
        
        
    def plot_3d_XYZ_lines(self):
        # plot the X,Y,Z lines around the 3D head
        max_num, interval = 20, 0.2
        cnt = int(max_num/interval)
        x = list(np.arange(0, max_num, interval)) + [0]*cnt*2
        y = [0]*cnt + list(np.arange(0, max_num, interval)) + [0]*cnt
        z = [0]*cnt*2 + list(np.arange(0, max_num, interval))
        s = [1] * cnt * 3
        self.visualization.scene.mlab.points3d(x[:cnt], y[:cnt], z[:cnt], s[:cnt], 
            color=(1,0,0), colormap="copper", scale_factor=0.2)  # red
        self.visualization.scene.mlab.points3d(x[cnt:2*cnt], y[cnt:2*cnt], z[cnt:2*cnt], s[cnt:2*cnt], 
            color=(0,1,0), colormap="copper", scale_factor=0.2)  # green
        self.visualization.scene.mlab.points3d(x[2*cnt:3*cnt], y[2*cnt:3*cnt], z[2*cnt:3*cnt], s[2*cnt:3*cnt], 
            color=(0,0,1), colormap="copper", scale_factor=0.2)  # blue

        four_pts = [[0,0,0], [max_num,0,0], [0,max_num,0], [0,0,max_num]]
        return four_pts
        

    def plot_3d_cube_ploygon(self):
        r, interval = 20, 0.2  # cube radius is 20
        # r, interval = 16, 0.4  # cube radius is 16
        cnt = int(r/interval)
        pt1, pt2, pt3, pt4 = [-r//2, 0, -r//4], [r//2, 0, -r//4], [r//2, r, -r//4], [-r//2, r, -r//4]
        pt5, pt6, pt7, pt8 = [-r//2, 0, r-r//4], [r//2, 0, r-r//4], [r//2, r, r-r//4], [-r//2, r, r-r//4]
        self.visualization.scene.mlab.points3d(list(np.arange(pt1[0], pt2[0], interval)), [pt1[1]]*cnt, [pt1[2]]*cnt, [1]*cnt, 
            color=(1,0,0), colormap="copper", scale_factor=0.2)  # line pt1 --> pt2
        self.visualization.scene.mlab.points3d([pt1[0]]*cnt, list(np.arange(pt1[1], pt4[1], interval)), [pt1[2]]*cnt, [1]*cnt, 
            color=(0,1,0), colormap="copper", scale_factor=0.2)  # line pt1 --> pt4
        self.visualization.scene.mlab.points3d([pt1[0]]*cnt, [pt1[1]]*cnt, list(np.arange(pt1[2], pt5[2], interval)), [1]*cnt, 
            color=(0,0,1), colormap="copper", scale_factor=0.2)  # line pt1 --> pt5
        
        self.visualization.scene.mlab.points3d([pt2[0]]*cnt, list(np.arange(pt2[1], pt3[1], interval)), [pt2[2]]*cnt, [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt2 --> pt3
        self.visualization.scene.mlab.points3d(list(np.arange(pt4[0], pt3[0], interval)), [pt3[1]]*cnt, [pt3[2]]*cnt, [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt4 --> pt3
            
        self.visualization.scene.mlab.points3d([pt8[0]]*cnt, list(np.arange(pt5[1], pt8[1], interval)), [pt8[2]]*cnt, [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt5 --> pt8
        self.visualization.scene.mlab.points3d(list(np.arange(pt5[0], pt6[0], interval)), [pt6[1]]*cnt, [pt6[2]]*cnt, [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt5 --> pt6
        self.visualization.scene.mlab.points3d([pt6[0]]*cnt, list(np.arange(pt6[1], pt7[1], interval)), [pt6[2]]*cnt, [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt6 --> pt7
        self.visualization.scene.mlab.points3d(list(np.arange(pt8[0], pt7[0], interval)), [pt8[1]]*cnt, [pt8[2]]*cnt, [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt8 --> pt7
            
        self.visualization.scene.mlab.points3d([pt2[0]]*cnt, [pt2[1]]*cnt, list(np.arange(pt2[2], pt6[2], interval)), [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt2 --> pt6
        self.visualization.scene.mlab.points3d([pt3[0]]*cnt, [pt3[1]]*cnt, list(np.arange(pt3[2], pt7[2], interval)), [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt3 --> pt7
        self.visualization.scene.mlab.points3d([pt8[0]]*cnt, [pt8[1]]*cnt, list(np.arange(pt4[2], pt8[2], interval)), [1]*cnt, 
            color=(0,0,0), colormap="copper", scale_factor=0.2)  # line pt4 --> pt8
        
        eight_pts = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]
        return eight_pts
   
    def plot_3d_refer_landmarks(self):
        
        pt0 = [0.2041, 9.1852, 13.6395]     # nose
        pt1 = [1.5098, 11.1738, 11.2761]    # left_inner_eye
        pt2 = [-1.1499, 11.2361, 11.2640]   # right_inner_eye
        pt3 = [3.3540, 11.3355, 10.5802]    # left_outer_eye
        pt4 = [-2.9746, 11.2949, 10.5292]   # right_outer_eye
        pt5 = [2.0159, 6.8906, 11.4766]     # left_mouth_corner
        pt6 = [-1.5811, 6.9059, 11.4898]    # right_mouth_corner
        pt7 = [0.2454, 4.1796, 12.1119]     # middle_chin
        pt8 = [4.9726, 8.3363, 6.6660]      # left_down_ear
        pt9 = [5.6806, 11.6000, 6.4259]     # left_up_ear
        pt10 = [-4.5943, 8.3982, 6.6922]    # right_down_ear
        pt11 = [-5.2043, 11.5301, 6.4909]   # right_up_ear
        pt12 = [0.1978, 9.6185, 0.3922]     # head_back_center
        pt13 = [0.1940, 19.1974, 6.5763]    # head_top_center
        
        refer_lms = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12, pt13]
        for pt_x, pt_y, pt_z in refer_lms:      
            self.visualization.scene.mlab.points3d(pt_x, pt_y, pt_z, 1, 
                color=(1,1,0), colormap="copper", scale_factor=0.6)  # yellow
        
        return refer_lms
        
    def update_head_pose_with_euler_angles(self, yaw, roll, pitch):
        '''
        This way of euler_angles --> head_pose is inspired by official mayavi interaction method by search "keyborad" and "up down".
        https://github.com/enthought/mayavi/blob/9d55fc4d6d482b02775aac0b6a528a4d73e2a673/tvtk/pyface/ui/qt4/scene.py
        https://github.com/enthought/mayavi/blob/9d55fc4d6d482b02775aac0b6a528a4d73e2a673/tvtk/pyface/ui/wx/scene.py
        '''
        
        """We could check the function using scripts in E:\dataset_public\HeadPose\FSANet_data\type1\*.py on ALFW2000"""
        
        # order: roll yaw pitch
        self.visualization.scene.mlab.view(*(self.init_view))
         
        self.visualization.scene.mlab.roll(-roll)  # for camera view, not object view

        self.visualization.scene.camera.azimuth(yaw)  # for object view, not camera view

        self.visualization.scene.camera.elevation(-pitch)  # for object view, not camera view
        self.visualization.scene.camera.orthogonalize_view_up()

        # _, _, fixed_distance, fixed_focalpoint = self.init_view
        # self.visualization.scene.mlab.view(distance=fixed_distance, focalpoint=fixed_focalpoint)

        print("Updated head pose [yaw, pitch, roll]:", yaw, pitch, roll)
        print("Updated view: ", self.visualization.scene.mlab.view())
        
        
    def mlab_3d_to_2d(self):
        '''
        Calculate the projection of 3D world coordinates to 2D display coordinates (pixel coordinates) for a given scene.
        '''
        # pts3d = np.array(self.eight_pts)  # shape [8,3]
        pts3d = np.array(self.eight_pts + self.refer_lms)  # shape [8+14,3]
        
        W = np.ones((pts3d.shape[0], 1))
        # hmgns_world_coords = np.hstack((pts3d, W))  # shape [8,4] 
        hmgns_world_coords = np.hstack((pts3d, W))  # shape [8+14,4] 
        
        comb_trans_mat = get_world_to_view_matrix(self.visualization.scene)
        view_coords = apply_transform_to_points(hmgns_world_coords, comb_trans_mat)
        
        # to get normalized view coordinates, we divide through by the fourth element
        norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))
        
        # the last step is to transform from normalized view coordinates to display coordinates.
        view_to_disp_mat = get_view_to_display_matrix(self.visualization.scene)
        disp_coords = apply_transform_to_points(norm_view_coords, view_to_disp_mat)
        
        # at this point, disp_coords is an Nx4 array of homogenous coordinates,
        # where X and Y are the pixel coordinates of the X and Y 3D world coordinates
        # return disp_coords[:, 0:2]  # [8,4] --> [8,2]
        return disp_coords[:, 0:2]  # [8+14,4] --> [8+14,2]
        

    def calculate_euler_angles(self):
        
        # eights_pts2d = self.mlab_3d_to_2d()
        eights_pts2d = self.mlab_3d_to_2d()[:8, :]
        eights_pts2d = list(eights_pts2d[::-1, :])
        
        '''we only need pt1, pt2, pt3 and pt7'''
        pt1x, pt1y = eights_pts2d[0][0], eights_pts2d[0][1]
        pt2x, pt2y = eights_pts2d[1][0], eights_pts2d[1][1]
        pt3x, pt3y = eights_pts2d[2][0], eights_pts2d[2][1]
        pt7x, pt7y = eights_pts2d[6][0], eights_pts2d[6][1]

        zero_float = 1e-15
        K1 = (pt1x - pt2x) / (zero_float if abs(pt1y - pt2y) < zero_float else (pt1y - pt2y))
        K2 = (pt3x - pt2x) / (zero_float if abs(pt3y - pt2y) < zero_float else (pt3y - pt2y))
        K3 = (pt3x - pt7x) / (zero_float if abs(pt3y - pt7y) < zero_float else (pt3y - pt7y))

        alpha = K3*K3*(K1*K2 + 1)
        beta = (K1-K3)*(K2-K3)
        
        '''yaw'''
        cos_y_value_real = 1-abs(2*alpha/beta)
        cos_y_value = np.clip(cos_y_value_real, -1, 1)
        yaw = math.acos(cos_y_value) / 2
        if pt1x < pt2x:
            # range [0, 90] & range [-90, 0]
            yaw = abs(yaw) if pt3x > pt7x else abs(yaw)*(-1)
        else:
            # range [90, 180] & range [-180, -90]
            yaw = -abs(yaw)+np.pi if pt3x > pt7x else abs(yaw)-np.pi

        '''pitch'''
        K3 = (zero_float if abs(K3) < zero_float else K3)
        sin_p_value_real = -math.tan(yaw)/K3
        sin_p_value = np.clip(sin_p_value_real, -1, 1)
        pitch = math.asin(sin_p_value)  # range [0, 90] & range [-90, 0]
        if pt2y > pt3y:  # range [90, 180] & range [-180, -90]
            pitch = np.pi-pitch if pitch > 0 else -np.pi-pitch


        '''roll'''
        # tan_r_value = (math.cos(yaw)/K1 - math.sin(pitch)*math.sin(yaw)) / math.cos(pitch)
        # tan_r_value = math.cos(pitch) / (math.sin(pitch)*math.sin(yaw) - math.cos(yaw)/K2)
        # roll = math.atan(tan_r_value)  # range [0, 90] & range [-90, 0]
        
        # roll = math.atan2(math.cos(yaw)/K1 - math.sin(pitch)*math.sin(yaw), math.cos(pitch))  # range [-180, 180]
        # roll = math.atan2(math.cos(pitch), math.sin(pitch)*math.sin(yaw) - math.cos(yaw)/K2)  # range [-180, 180]
        
        tan_r_value_pow_up = math.cos(yaw)/K1 - math.sin(pitch)*math.sin(yaw)
        tan_r_value_pow_down = math.sin(pitch)*math.sin(yaw) - math.cos(yaw)/K2
        tan_r_value_pow_down = zero_float if abs(tan_r_value_pow_down) < zero_float else tan_r_value_pow_down
        tan_r_value = math.sqrt(abs(tan_r_value_pow_up / tan_r_value_pow_down))
        roll = math.atan(tan_r_value)  # range [0, 90]
        if pt1x < pt2x:
            if pt2x > pt3x:
                # range [90, 180] & range [0, 90]
                roll = np.pi-roll if pt1x > pt2x else roll
            else:
                # range [-180,-90] & range [-90, 0]
                roll = roll-np.pi if pt1x > pt2x else -roll
        else:
            if pt2x < pt3x:
                # range [90, 180] & range [0, 90]
                roll = np.pi-roll if pt1x < pt2x else roll
            else:
                # range [-180,-90] & range [-90, 0]
                roll = roll-np.pi if pt1x < pt2x else -roll
                

        '''generate final outputs'''
        yaw = -(yaw / np.pi * 180)
        roll = roll / np.pi * 180
        pitch = pitch / np.pi * 180
        
        
        '''make all anlges are odd numbers'''
        yaw = int(2 * (yaw // 2) + 1) if yaw != 180 else 179
        roll = int(2 * (roll // 2) + 1) if roll != 180 else 179
        pitch = int(2 * (pitch // 2) + 1) if pitch != 180 else 179
        

        print("\n calculated [yaw, pitch, roll]: \t", yaw, pitch, roll)
        
        print("mayavi roll: ", self.visualization.scene.mlab.roll())  # for camera view, not object view
        print("mayavi view(): ", self.visualization.scene.mlab.view())  # for camera view, not object view

        
        
        return yaw, roll, pitch
        