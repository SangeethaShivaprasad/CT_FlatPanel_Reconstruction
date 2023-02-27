import flat_panel_project_utils as utils
from Grid import Grid
import math
import numpy as np
import matplotlib.pyplot as plt


class parallelBeam:
    def create_sinogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_scan_range):
        # number_of_projections = total number of times the detector is rotated
        detLength2D = detector_sizeInPixels * detector_spacing
        delTheta = angular_scan_range / number_of_projections

        gridObject = Grid(detector_sizeInPixels, number_of_projections, [detector_spacing, delTheta])

        for i in range(number_of_projections):
            cosTheta = math.cos(math.radians(delTheta * i))
            sinTheta = math.sin(math.radians(delTheta * i))
            for j in range(detector_sizeInPixels):
                # the distance from the center
                #s = detector_spacing * j - detLength2D / 2.0
                s = detector_spacing * j - detLength2D / 2.0 + detector_spacing / 2.0 #pixel center
                p = 0

                #upper_bound = int(0.75 * phantom.get_size()[0])
                upper_bound = int(0.75 * phantom.get_size()[0] * phantom.get_spacing()[0])
                bound_list = np.arange(-upper_bound, upper_bound + phantom.get_spacing()[0], phantom.get_spacing()[0])

                #for r in range(-upper_bound, upper_bound, 1): #use phantom spacing
                for r in bound_list:
                    xPoints = r * cosTheta - s * sinTheta
                    yPoints = r * sinTheta + s * cosTheta
                    x, y = phantom.physical_to_index(xPoints, yPoints)
                    # p = p + utils.interpolate(phantom, xPoints, yPoints) * gridObject.get_spacing()[0]
                    p = p + utils.interpolate(phantom, x, y)
                gridObject.set_at_index(j, i, p)

        return gridObject

    def backproject(sinogram, reco_size_x, reco_size_y, spacing):
        num_projections = sinogram.get_size()[1]
        dtheta = sinogram.get_spacing()[1]
        reco_img = Grid(reco_size_y, reco_size_x, spacing)
        for k in range(num_projections):
            for i in range(reco_size_y):
                for j in range(reco_size_x):
                    x_p = reco_img.index_to_physical(j, i)[0]
                    y_p = reco_img.index_to_physical(j, i)[1]
                    theta = math.radians(k * dtheta)
                    #s = ((x_p * math.cos(theta)) + (y_p * math.sin(theta)))  # check
                    s = ((x_p * math.cos(theta)) - (y_p * math.sin(theta)))
                    x, y = sinogram.physical_to_index(s, theta)
                    back_ray_val = utils.interpolate(sinogram, x, k)
                    # back_ray_val = back_ray_val / (reco_img.get_size()[1] * reco_img.get_spacing()[0] * math.sqrt(2) * sinogram.get_size()[1])
                    pix = reco_img.get_at_index(i, j) + back_ray_val
                    reco_img.set_at_index(i, j, pix)
        return reco_img

    def next_power_of_two(value):
        value = value - 1
        while value & value - 1:
            value = value & value - 1
        # return next power of 2 multiplied by 2
        #print((value << 1) * 2)
        return (value << 1) * 2


    def ramp_filter(sinogram, detector_spacing):
        detector_size, num_projections = sinogram.get_size()
        next_power_2 = parallelBeam.next_power_of_two(sinogram.get_size()[0])

        pad_value = int((next_power_2 - sinogram.get_size()[0])/2)
        zero_pad_matrix = np.zeros((pad_value, num_projections))

        padded_sinogram = np.append(zero_pad_matrix, np.append(sinogram.get_buffer(), zero_pad_matrix, axis=0), axis=0)
        filtered_sinogram = np.zeros(padded_sinogram.shape)
        frequency_spacing = 1 / (detector_spacing * next_power_2)

        w_bounds = (next_power_2 * frequency_spacing) / 2
        w = np.arange(-w_bounds, w_bounds, frequency_spacing)
        ramp_filter = np.abs(w)

        ramp_filter_corrected = np.fft.fftshift(ramp_filter)

        for i in range(num_projections):
            projection_fft = np.fft.fft(padded_sinogram[:, i])
            filtered_projection_shifted = projection_fft * ramp_filter_corrected
            filtered_sinogram[:, i] = np.real(np.fft.ifft(filtered_projection_shifted))
        gridObject = Grid(filtered_sinogram.shape[0], filtered_sinogram.shape[1], sinogram.get_spacing())
        gridObject.set_buffer(filtered_sinogram)
        return gridObject

    def ramlak_filter(sinogram, detector_spacing):
        detector_size, num_projections = sinogram.get_size()
        next_power_2 = parallelBeam.next_power_of_two(sinogram.get_size()[0])

        pad_value = int((next_power_2 - sinogram.get_size()[0])/2)
        zero_pad_matrix = np.zeros((pad_value, num_projections))

        padded_sinogram = np.append(zero_pad_matrix, np.append(sinogram.get_buffer(), zero_pad_matrix, axis=0), axis=0)
        filtered_sinogram = np.zeros(padded_sinogram.shape)

        frequency_spacing = 1 / (detector_spacing * next_power_2)
        w_bounds = (next_power_2 * frequency_spacing) / 2
        w = np.arange(-w_bounds, w_bounds, frequency_spacing)
        # todo - should I shift w so that negative n positive axis are swapped?

        ramlak_filter = np.zeros(len(w))
        ramp_w_bound = int(len(w)/2)

        for n in range(-ramp_w_bound, ramp_w_bound):
            if n == 0:
                h_n = 1/ (4 *(detector_spacing ** 2))
                ramlak_filter[n] = h_n
            elif (n % 2) == 0 :
                h_n = 0
                ramlak_filter[n] = h_n
            else:
                h_n = -1/(((n * detector_spacing) ** 2) * (np.pi ** 2))
                #ramlak_filter.append(h_n)
                ramlak_filter[n] = h_n

        #utils.show(ramlak_filter, "ramlak_before")
        ramlak_filter_fft = np.fft.fft(ramlak_filter)
        #utils.show(ramlak_filter_fft, "ramlak_after")

        for i in range(num_projections):
            projection_fft = np.fft.fft(padded_sinogram[:, i])
            filtered_projection_fft = projection_fft * ramlak_filter_fft
            filtered_sinogram[:, i] = np.real(np.fft.ifft(filtered_projection_fft))
        gridObject = Grid(filtered_sinogram.shape[0], filtered_sinogram.shape[1], sinogram.get_spacing())
        gridObject.set_buffer(filtered_sinogram)
        return gridObject