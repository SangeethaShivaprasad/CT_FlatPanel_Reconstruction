import flat_panel_project_utils as utils
from Grid import Grid
import math
import numpy as np
import matplotlib.pyplot as plt
class FanBeam:
    def create_fanogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_increment, d_si, d_sd):

        grid_object = Grid(detector_sizeInPixels, number_of_projections, [detector_spacing, angular_increment])
        d_id = d_sd - d_si # center to detector distance
        origin = -((detector_sizeInPixels - 1)/2 * detector_spacing)

        for i in range(number_of_projections):
            b = angular_increment * i #calculated beta
            cosb = math.cos(math.radians(b))
            sinb = math.sin(math.radians(b))
            s = (-d_si * sinb, d_si * cosb) #the source coordinate
            m = (d_id * sinb, -d_id * cosb) #mid point of the detector

            for j in range(detector_sizeInPixels):
                t = j * detector_spacing + origin

                # get the physical position wrt its origin
                mp = (t * cosb, t * sinb)
                p = (m[0] + mp[0], m[1] + mp[1])
                sp_x = p[0] - s[0]
                sp_y = p[1] - s[1]
                sp = math.sqrt((math.pow(sp_x, 2)) + (math.pow(sp_y, 2)))
                #get the unit vector
                uv_x = sp_x/sp
                uv_y = sp_y/sp

                range_list = np.arange(0, sp, phantom.get_spacing()[0])
                ray_value = 0

                for r in range_list:
                    x_pt = s[0] + r * uv_x
                    y_pt = s[1] + r * uv_y
                    #x, y = phantom.physical_to_index(x_pt, y_pt)
                    ray_value = ray_value + phantom.get_at_physical(x_pt, y_pt)

                sampling_value = ray_value * phantom.get_spacing()[0]
                grid_object.set_at_index(j, i, sampling_value)

                #grid_object.set_at_index(j, i, ray_value)
        #test = grid_object.get_buffer()
        #test1 = test[:, :min_angular_scan_range]
        #grid_object.set_buffer(test1)

        return grid_object

    """def rebinning(fanogram, d_si, d_sd):
        detector_sizeInPixels = fanogram.get_Size()[0]
        number_of_projections = fanogram.get_Size()[1]
        detector_spacing = fanogram.get_spacing()[0]
        beta = fanogram.get_spacing()[1]

        sinogram = Grid(detector_sizeInPixels, number_of_projections, [detector_spacing, beta])"""


    def rebinning(fanogram, d_si, d_sd):

        num_proj_beta = fanogram.get_buffer().shape[1]
        detector_sizeInPixels = fanogram.get_size()[0]
        det_spacing = fanogram.get_spacing()[0]
        min_angular_scan_range = num_proj_beta * fanogram.get_spacing()[1]
        origin = -(((detector_sizeInPixels - 1) / 2) * det_spacing)
        delta_theta = 180 / num_proj_beta
        # gridObject = Grid(detector_sizeInPixels, num_proj_beta, [det_spacing, fanogram.get_spacing()[1]])
        gridObject = Grid(detector_sizeInPixels, num_proj_beta, [det_spacing, delta_theta])
        gridObject.set_origin((origin, 0))
        fanogram.set_origin((origin, (-(0.5 * (num_proj_beta - 1) * delta_theta))))
        for th in range(num_proj_beta):
            for s in range(detector_sizeInPixels):  # s
                s_new = gridObject.index_to_physical(s, 0)[0]
                # theta = th * (fanogram.get_spacing()[1]) # degrees
                theta = th * delta_theta  # degrees
                gamma = math.degrees(math.asin(s_new / d_si))  # deg
                t = d_sd * math.tan(math.radians(gamma))
                beta = theta - gamma  # deg
                if (beta > min_angular_scan_range):
                    gamma_two = - gamma
                    beta_two = beta + (2 * gamma_two) - 180  # rad
                    #beta_two = beta - (2 * gamma_two) + 180  # rad
                    t_2 = -t
                    set_value = fanogram.get_at_physical(t_2, beta_two)
                    gridObject.set_at_index(s, th, set_value)
                elif (beta < 0):
                    gamma_two = - gamma
                    beta_two = beta - (2 * gamma_two) + 180  # rad
                    #beta_two = beta + (2 * gamma_two) - 180  # rad
                    t_2 = -t
                    set_value = fanogram.get_at_physical(t_2, beta_two)
                    gridObject.set_at_index(s, th, set_value)
                else:
                    set_value = fanogram.get_at_physical(t, beta)
                    gridObject.set_at_index(s, th, set_value)

        return gridObject