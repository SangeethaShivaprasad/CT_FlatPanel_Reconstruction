from Grid import Grid
import flat_panel_project_utils as utils
from parallelBeamProjection import parallelBeam
from FanbeamProjection import FanBeam
import math

shepp = utils.shepp_logan(64)
shepp_grid = Grid(64,64, [0.5,0.5])

#set buffer
shepp_grid.set_buffer(shepp)
utils.show(shepp, "original_image")

#create_sinogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_scan_range)
# sinogram = parallelBeam.create_sinogram(shepp_grid, 360, 0.5, 140, 180)
# utils.show(sinogram.get_buffer(), "Sinogram")
#
# back_projection = parallelBeam.backproject(sinogram, 128, 128, [0.5, 0.5])
# utils.show(back_projection.get_buffer(), "Back_Projection")
#
# ramp_filter = parallelBeam.ramp_filter(sinogram, 0.5)
# utils.show(ramp_filter.get_buffer(), "Ramp_filter")
#
# back_projection = parallelBeam.backproject(ramp_filter, 128, 128, [0.5, 0.5])
# utils.show(back_projection.get_buffer(), "Ramp_filter_BP")
#
# ramLak_filter = parallelBeam.ramlak_filter(sinogram, 0.5)
# utils.show(ramLak_filter.get_buffer(), "RamLak_filter")
#
# back_projection = parallelBeam.backproject(ramLak_filter, 128, 128, [0.5, 0.5])
# utils.show(back_projection.get_buffer(), "Ramlak_filter_BP")

detector_sizeInPixels = 250
detector_spacing = 0.5
angular_increment = 2
d_si = 140
d_sd = 280

fan_angle_gamma = int(math.degrees(math.atan((detector_sizeInPixels * detector_spacing) / d_sd)))  # gamma
min_angular_scan_range = 180 + fan_angle_gamma
number_of_projections = int(min_angular_scan_range / angular_increment)

Fanogram = FanBeam.create_fanogram(shepp_grid, number_of_projections, detector_spacing, detector_sizeInPixels, angular_increment, d_si, d_sd)
#Fanogram = FanBeam.create_fanogram(shepp_grid, 360, 0.5, 250, 1, 140, 280)
#Fanogram = FanBeam.create_fanogram(shepp_grid, 45, 0.5, 50, 8, 50, 100)
utils.show(Fanogram.get_buffer(), "Fanogram")

#rebinning(fanogram, d_si, d_sd)
rebinning = FanBeam.rebinning(Fanogram, d_si, d_sd)
utils.show(rebinning.get_buffer(), "Rebinning")

ramLak_filter = parallelBeam.ramlak_filter(rebinning, 0.5)
utils.show(ramLak_filter.get_buffer(), "RamLak_filter")

fanogram_back_projection = parallelBeam.backproject(ramLak_filter,64, 64,[0.5,0.5])
utils.show(fanogram_back_projection.get_buffer(), "Ramlak_filter_BP")

#plt.imshow(sinogram, cmap="gray")
#plt.show()