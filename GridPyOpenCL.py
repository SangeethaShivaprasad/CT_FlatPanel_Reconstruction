import numpy as np
import pyopencl as cl
from Grid import Grid
import flat_panel_project_utils as utils
import math
from timeit import default_timer as timer

from parallelBeamProjection import parallelBeam


def addGrid(gridA, gridB):
    shp = gridA.shape
    gridC = np.zeros_like(gridA)
    for i in range(shp[0]):
        for j in range(shp[1]):
            gridC[i][j] = gridA[i][j] + gridB[i][j]

    return gridC


def backprojectOpenCL(sinogram,size_x,size_y,spacing):
    num_projections = sinogram.get_size()[1]
    dtheta = sinogram.get_spacing()[1]
    reco_img = Grid(size_y, size_x, spacing)
    sino_grid = sinogram.get_buffer().astype(np.float32)
    sino_grid = np.ascontiguousarray(sino_grid, dtype=np.float32)
    reco_img_grid = reco_img.get_buffer().astype(np.float32)
    reco_img_grid = np.ascontiguousarray(reco_img_grid, dtype=np.float32)
    res_img = reco_img_grid.copy().astype(np.float32)
    res_img = np.ascontiguousarray(res_img, dtype=np.float32)

    # Load kernel and compile cl file
    platform = cl.get_platforms()
    gpu = platform[0].get_devices()

    ctx = cl.Context(gpu)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    kernel = open("GridKernel.cl").read()
    prg = cl.Program(ctx, kernel).build()

    # Create new GPU image buffers
    # Copy the host CPU buffers to GPU image buffers
    # Sinogram and current reconstructed image should be read only
    src_buf_sin = cl.image_from_array(ctx, sino_grid, 1, 'r')
    src_buf_rec = cl.image_from_array(ctx, reco_img_grid, 1, 'r')
    # Result Backproject image should be write only
    src_buf_res = cl.image_from_array(ctx, res_img, 1, 'w')

    # Call the GPU backproject device kernel function
    prg.backproject(queue,reco_img_grid.shape , None, src_buf_sin, np.int32(sinogram.get_origin()[0]),
                    np.float32(sinogram.get_spacing()[0]), src_buf_rec, np.int32(reco_img.get_origin()[0]),
                    np.int32(reco_img.get_origin()[1]), np.float32(spacing[0]), np.int32(num_projections), np.float32(dtheta),
                    src_buf_res)

    # Copy the result image from device GPU OpenCl buffer to Host CPU buffer
    cl.enqueue_copy(queue, res_img, src_buf_res, origin=(0, 0), region=res_img.shape[1::-1])
    reco_img.set_buffer(res_img)
    return reco_img


def testBackproject():
    shepp = utils.shepp_logan(128)
    shepp_grid = Grid(128, 128, [0.5, 0.5])
    shepp_grid.set_buffer(shepp)
    sinogram = parallelBeam.create_sinogram(shepp_grid, 180, 0.5, 140, 180)
    utils.show(sinogram.get_buffer(), "Sinogram")
    filtered_sinogrm = parallelBeam.ramp_filter(sinogram, sinogram.get_spacing()[0])
    utils.show(sinogram.get_buffer(), "Filtered Sinogram")

    start_bpgpu = timer()
    #back_projection = backprojectOpenCL(sinogram, 128, 128, [0.5, 0.5])
    back_projection = backprojectOpenCL(filtered_sinogrm, 128, 128, [0.5, 0.5])
    end_bpgpu = timer()
    print("GPU Back Projection execution")
    print(end_bpgpu - start_bpgpu)
    utils.show(back_projection.get_buffer(), "GPU Back_Projection")

    start_bpcpu = timer()
    #bck_proj = parallelBeam.backproject(sinogram, 128, 128, [0.5, 0.5])
    bck_proj = parallelBeam.backproject(filtered_sinogrm, 128, 128, [0.5, 0.5])
    end_bpcpu = timer()
    print("CPU Back Projection execution")
    print(end_bpcpu - start_bpcpu)
    utils.show(bck_proj.get_buffer(), "CPU Back_Projection")



print("Run back project")
testBackproject()
print("Run end")

shepp = utils.shepp_logan(128)
shepp_grid = Grid(128, 128, [0.5, 0.5])

#set buffer
shepp_grid.set_buffer(shepp)
#utils.show(shepp, "original_image")

# Prepare grid A,B and C
grid_A = shepp_grid.get_buffer().astype(np.float32)
utils.show(grid_A, "Original Image")
grid_dim = grid_A.shape
dim = grid_dim[0]*grid_dim[1]

# Rotate the array by 90 degree
grid_B = np.rot90(grid_A).astype(np.float32)
grid_B = np.ascontiguousarray(grid_B, dtype=np.float32)
grid_dim_B = grid_B.shape
utils.show(grid_B, "Grid B")

# Result image holder
grid_C = np.zeros_like(grid_A)

# Flatten the 2 grids
#A_1d = grid_A.flatten().astype(np.float32)
#print(A_1d.shape)
#B_1d = grid_B.flatten().astype(np.float32)
#C_1d = np.zeros_like(A_1d)

# Prepare to call opencl kernel
platform = cl.get_platforms()
gpu = platform[0].get_devices()

ctx = cl.Context(gpu)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Convert A_1d and B_1d to openCL array
#a_g = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=A_1d)
#b_g = cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=B_1d)
#c_g = cl.Buffer(ctx, mf.WRITE_ONLY, C_1d.nbytes)

#use 2D image opencl image
src_buf_A = cl.image_from_array(ctx, grid_A, 1,'r')
src_buf_B = cl.image_from_array(ctx, grid_B, 1,'r')
src_buf_C = cl.image_from_array(ctx, grid_C, 1,'w')


#test = np.random.rand(dim).astype(np.float32)
#print(test.shape)

# Load kernel and compile cl file
kernel = open("GridKernel.cl").read()
prg = cl.Program(ctx, kernel).build()
print("Kernel built")

start_g = timer()
#prg.gridAdd(queue, A_1d.shape, None, a_g, b_g) #grid add using flattened 1d array
prg.imgGridAdd(queue, grid_A.shape, None, src_buf_A, src_buf_B, src_buf_C) #grid add using openCL image
end_g = timer()
print("GPU execution")
print(end_g - start_g)

#cl.enqueue_copy(queue, C_1d, a_g)
cl.enqueue_copy(queue, grid_C, src_buf_C, origin=(0, 0), region=grid_C.shape[1::-1])

#grid_C = C_1d.reshape(grid_dim)
utils.show(grid_C, "Grid C")

#D_1d = A_1d + B_1d
#grid_D = D_1d.reshape(grid_dim)#grid_A + grid_B

# For loop execution of grid add
start_f = timer()
grid_D = addGrid(grid_A, grid_B)
end_f = timer()
print("For loop  execution")
print(end_f - start_f)
utils.show(grid_D, "Grid D")


