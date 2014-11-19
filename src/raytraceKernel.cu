// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/scan.h>


#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"




void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}



///////////////////////////////////
//////////////////////////////////
// TODO: IMPLEMENT THIS FUNCTION/
////////////////////////////////
///////////////////////////////
// Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, float DOF, float aperature){
  int index = x + (y * resolution.x);
  
  
  glm::vec3 alpha, beta, midPix, horizScale, vertScale, pixel;
  alpha  = glm::cross(view, up);
  beta   = glm::cross(alpha, view);
  midPix = eye + view;

  vertScale  = glm::normalize(beta)  * glm::length(view) * tan(glm::radians( - fov.y)); //had to flip this (it was upside down)
  horizScale = glm::normalize(alpha) * glm::length(view) * tan(glm::radians(fov.x));
  
  //jitter the pixel
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(-0.5,0.5);
  thrust::uniform_real_distribution<float> u02(-0.01,0.01);

  
  pixel = midPix + horizScale * (float)((2.0 * (x + (float)u01(rng))/resolution.x) - 1.0) + vertScale * (float)((2.0 * (y + (float)u01(rng))/resolution.y) - 1.0);
  ray r;
  
  //COMMENT OUT FOR DOF
  r.origin = eye;
  r.direction = glm::normalize(pixel - eye);
  
  /*   //UNCOMMENT FOR DOF
  r.origin = pixel;
  float aperatureOffsetX = (float)u01(rng) * aperature;//for DOF
  float aperatureOffsetY = (float)u01(rng) * aperature;//for DOF
  glm::vec3 focalDirection = glm::normalize(pixel - eye);
  glm::vec3 focalPoint = eye + (focalDirection * DOF);//for depth of field
  r.origin = r.origin + horizScale * (aperatureOffsetX/resolution.x) + vertScale * (aperatureOffsetY/resolution.y);
  r.direction = glm::normalize(focalPoint - r.origin);
  */
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//Initialize rays
__global__ void initializeRay(glm::vec2 resolution, float time, cameraData cam, rayState* rayList){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
    ray thisRay = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov, cam.DOF, cam.APERATURE);
    rayList[index].RAY      = thisRay;
    rayList[index].isValid  = 1;
    rayList[index].color    = glm::vec3(1,1,1);
    rayList[index].photoIDX = index;
  }
}



///////////////////////////////////
//////////////////////////////////
// TODO: IMPLEMENT THIS FUNCTION/ 
//   raytraceRay() should take in a camera, image buffer, geometry, materials, and lights, 
//   and should trace a ray through the scene and write the resultant color to a pixel in the image buffer.
////////////////////////////////
///////////////////////////////
// Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int maxDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
                            rayState* rayList, int currDepth, int* validRays, int length){
  //need to update for string compaction
  //int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  //int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  //int index = x + (y * resolution.x);
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index < length){
  //if((x<=resolution.x && y<=resolution.y)){
    if(rayList[index].isValid == 0){
      return;
    }
    if(currDepth >= maxDepth){//exceeded max depth
       //this contribution is black
      colors[rayList[index].photoIDX] = (colors[rayList[index].photoIDX] * (time - 1.0f)/time) + (glm::vec3(0,0,0) * 1.0f/time);
      rayList[index].isValid = 0;
      validRays[index] = 0;
      return;
    }
    //get variables
    ray thisRay     = rayList[index].RAY;
    glm::vec3 COLOR = rayList[index].color;

    //intersection checks:
    float distToIntersect = FLT_MAX;//infinite distance
    float tmpDist;
    glm::vec3 tmpIntersectPoint, tmpIntersectNormal, intersectPoint, intersectNormal;
    material mat;
    
    for(int i = 0; i < numberOfGeoms; i++){
      if (geoms[i].type == SPHERE){
        tmpDist = sphereIntersectionTest(geoms[i], thisRay, tmpIntersectPoint, tmpIntersectNormal);
      }else if (geoms[i].type == CUBE){
        tmpDist = boxIntersectionTest(   geoms[i], thisRay, tmpIntersectPoint, tmpIntersectNormal);
      }//insert triangles here for meshes
      if (tmpDist != -1 && tmpDist < distToIntersect){ //hit is new closest
        distToIntersect = tmpDist;
        intersectNormal = tmpIntersectNormal;
        intersectPoint  = tmpIntersectPoint;
        mat = materials[geoms[i].materialid];
      }
    }
    //Did I intersect anything?
    if(distToIntersect == FLT_MAX){//miss
      //this contribution is black
      colors[rayList[index].photoIDX] = (colors[rayList[index].photoIDX] * (time - 1.0f)/time) + (glm::vec3(0,0,0) * 1.0f/time);
      rayList[index].isValid = 0;
      validRays[index] = 0;
    }
    //is this a light source?
    if(mat.emittance > 0.001){
      COLOR = COLOR * (mat.color * mat.emittance);
      colors[rayList[index].photoIDX] = (colors[rayList[index].photoIDX] * (time - 1.0f)/time) + (COLOR * 1.0f/time);
      rayList[index].isValid = 0;
      validRays[index] = 0;
      return;
    }
    
    //update variables
    thrust::default_random_engine rng(hash(index * (time + currDepth)));
    thrust::uniform_real_distribution<float> u01(0,1);
    calculateBSDF(thisRay, intersectPoint, intersectNormal, COLOR, mat, (float) u01(rng) ,(float) u01(rng)); 
    //update struct
    rayList[index].RAY   = thisRay;
    rayList[index].color = COLOR;
  }
}

__global__ void compactRays(int* scanRays, rayState* rayList, int* validRays, int length){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index >= length){
    return;
  }
  validRays[index] = 0;
  if(index == 0){//first 
    return;
  }

  if(scanRays[index - 1] < scanRays[index]){
    rayState newRay = rayList[index];
    __syncthreads();
    rayList[scanRays[index]] = newRay;
    validRays[scanRays[index]] = 1;
  }
}


///////////////////////////////////
//////////////////////////////////
// TODO: Finish THIS FUNCTION /// You will have to complete this function to support passing materials and lights to CUDA
////////////////////////////////
///////////////////////////////
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 10; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  // send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage,           (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms,   numberOfGeoms * sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms * sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  // package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;
  cam.DOF = renderCam->DOF[frame];//new
  cam.APERATURE = renderCam->APERATURE[frame];//new
  
  // package materials
  material* materialList = NULL;
  cudaMalloc((void**) &materialList,   numberOfMaterials * sizeof(material));
  cudaMemcpy( materialList, materials, numberOfMaterials * sizeof(material), cudaMemcpyHostToDevice);
  
  //allocate Rays
  rayState* rayList = NULL;
  cudaMalloc((void**)&rayList, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(rayState));

  


  // kernel launches
  //Get initial rays
  initializeRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, rayList);
  thrust::device_vector<int> validRays((int)renderCam->resolution.x * (int)renderCam->resolution.y, 1);
  int* thrustArray = thrust::raw_pointer_cast( &validRays[0] );
  int length = thrust::count(validRays.begin(), validRays.end(), 1);//count valid rays
  thrust::device_vector<int> scanRay((int)renderCam->resolution.x * (int)renderCam->resolution.y, 0);
  int* scanPointer = thrust::raw_pointer_cast( &scanRay[0] );
  
  
  //depth trace with compaction
  for(int i = 0; i <= traceDepth; i++){
    //do one step
    raytraceRay<<<(int)ceil((float)length/64.0f), 64>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, materialList, numberOfMaterials, rayList, i, thrustArray, length);
    //build scan
    thrust::exclusive_scan(validRays.begin(), validRays.end(), &scanRay[0]);
    scanPointer = thrust::raw_pointer_cast( &scanRay[0] );
    //compact rays
    compactRays<<<(int)ceil((float)length/64.0f), 64 >>>(scanPointer, rayList, thrustArray, length);
    //update length
    length = thrust::count(validRays.begin(), validRays.end(), 1);//count valid rays
  }

  //update visual
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree(materialList); //added
  cudaFree(rayList); //added
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
