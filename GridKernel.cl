
kernel void gridAdd(global float *A, global float *B)
{
    int gid = get_global_id(0);

    A[gid] += B[gid];
}

kernel void imgGridAdd(read_only image2d_t A, read_only image2d_t B, write_only image2d_t C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_image_width(A);
    int height = get_image_height(A);

    int2 pixelcoord = (int2) (get_global_id(0), get_global_id(1));

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP;

    if (pixelcoord.x < width && pixelcoord.y < height)
    {
    float4 pixelA = read_imagef(A, sampler, (int2)(pixelcoord.x, pixelcoord.y));
    float4 pixelB = read_imagef(B, sampler, (int2)(pixelcoord.x, pixelcoord.y));

    float4 pixelC = pixelA + pixelB;
    write_imagef(C, pixelcoord, pixelC);
    }
}


// Back projection implementation in GPU
kernel void backproject(read_only image2d_t sinogram, int org_xs, float spc_xs, read_only image2d_t reco, int org_x, int org_y, float spacing, int numproj, float dtheta, write_only image2d_t result)
{
    // Get the coordinate of the reconstruction image pixel
    int i = get_global_id(0);
    int j = get_global_id(1);


    // Calculate the physical coordinate of reconstructed image pixel
    int x_p = org_x + j*spacing;
    int y_p = org_y + i*spacing;

    //sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // create pixel coordinate of recostructed image
    int2 pixelcoord_reco = (int2) (j,i);
    //float4 pixel_res = read_imagef(reco, sampler, (int2)(pixelcoord_reco.x, pixelcoord_reco.y));
    float4 pixel_res = (float4)(0,0,0,0);

    // Loop for all projections
    for(int k = 0; k < numproj; ++k)
    {
    // calculate the angle of current projection
    float theta = radians(k*dtheta);

    // Calculate S, distance from origin
    float xcos = x_p * cos(theta);
    float ysin = y_p * sin(theta);
    float s = xcos - ysin;
    //float s = xcos + ysin;
    float x_s = (s - org_xs) / spc_xs; // calculate the x cordinate of sinogram pixel

    // create pixel coordinate of sinogram image
    float2 pixelcoord_sin = (float2) (k, x_s);
    pixelcoord_sin = pixelcoord_sin + (float2)(0.0f,0.5f); //shift the pixel by 0.5

    // Read pixel value of sinogram and current reconstruction
    float4 pixel_sin = read_imagef(sinogram, sampler, (int2)(pixelcoord_sin.x, pixelcoord_sin.y));


    // Calculate new pixel value and save it to result pixel
    pixel_res = pixel_res + pixel_sin;
    }

    //write the value to the
    write_imagef(result, pixelcoord_reco, pixel_res.x);
}

