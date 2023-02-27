import pyconrad
import math
import numpy as np
import flat_panel_project_utils as utils
from Grid import Grid


def create_sinogram(phantom, number_of_projections,detector_spacing, detector_sizeInPixels, angular_scan_range):
    sinogram = Grid(number_of_projections, detector_sizeInPixels, detector_spacing)
    sinogram.set_origin()
    sinOrg = sinogram.get_origin()
    for iAng in range(number_of_projections):
        for iW in range(detector_sizeInPixels):
            s = (iW*detector_spacing) + sinOrg[1]
            theta = iAng*(angular_scan_range/number_of_projections)
            cos = math.cos(math.radians(theta))
            sin = math.sin(math.radians(theta))
            R = 0.75*phantom.get_size()[1] # As given in execise sheet
            p = 0
            for r in range(-R,R,1):
                x = r*cos - s*sin
                y = r*sin + s*cos
                p = p + utils.interpolate(phantom, x,y)*phantom.get_spacing()
            sinogram.set_at_index(iAng,iW, p)










    

