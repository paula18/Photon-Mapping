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

__global__ void initializeLightPaths(float time, cameraData cam, rayState* lightrayList, int numLightpaths, staticGeom* lights, int numLights, Path* lightPaths, material* materialList){
	/*
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if((x<=cam.resolution.x && y<=cam.resolution.y))
	*/
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if((index <= numLightpaths))
	{
		//generate random numbers
		thrust::default_random_engine rng(hash(index * time));
		thrust::uniform_real_distribution<float> u01(-1,1);
		thrust::uniform_real_distribution<float> u02(0,1);
		float random  = (float) u02(rng) * 2.0 - 1.0;
		float random2 = (float) u02(rng) * 2.0 - 1.0;

		//randomly select light
		staticGeom light = lights[0];


		ray thisRay;
		glm::vec3 normal = getRandomDirectionInSphere(random, random2, glm::vec3(0,0,0)); // random point on unit sphere
		thisRay.origin = multiplyMV(light.transform, glm::vec4(normal,1.0f));
		random  = (float) u02(rng);
		random2 = (float) u02(rng);
		thisRay.direction = calculateRandomDirectionInHemisphere(normal, random, random2);

		//lightrayList[index].color = glm::vec3(3.0);
		lightrayList[index].RAY = thisRay;
		lightrayList[index].isValid = 1;

		//initialize light Path
		vertex v;
		v.isValid = 1;//This is the light!
		v.hitLight = 1;
		v.mat = materialList[light.materialid];
		v.outDirection = glm::vec3(0);
		v.inDirection = - thisRay.direction;
		v.pdfWeight = 1.0f;
		v.position = thisRay.origin;
		lightPaths[index].vert[0] = v;

	}
}

__host__ __device__ float getSolidAngle(staticGeom light, glm::vec3 position, glm::vec3 normal){
	
	glm::vec3 p1 = glm::normalize(glm::vec3(1,1,1)); //point on a unit sphere
	glm::vec3 pOnSphere = multiplyMV(light.transform, glm::vec4(p1,1.0f));//point on our sphere
	glm::vec3 centerOfSphere = multiplyMV(light.transform, glm::vec4(0,0,0,1.0f));// center of sphere
	
	glm::vec3 direction = centerOfSphere - position; 
	
	float radius = glm::distance(pOnSphere, centerOfSphere);
	float dist = glm::length(direction); 
	float angle = glm::atan(radius/dist);
	float solid = TWO_PI * (1.0f - glm::cos(angle));
	return solid *(1.0/PI);
	//Convert to PDFWeight
	//return solid * (dist * dist) / abs(glm::dot(normal,direction))/TWO_PI;
}

__host__ __device__ float convertSolidAngle(float value, float dist, float cos){
	return value * dist * dist / abs(cos); 

}
__host__ __device__ float intersectionTest(staticGeom* geoms, int numberOfGeoms, int& materialID, ray thisRay, float distToIntersect, glm::vec3& intersectNormal, glm::vec3& intersectPoint){
  float tmpDist;
  glm::vec3 tmpIntersectPoint, tmpIntersectNormal;
  for(int i = 0; i < numberOfGeoms; i++){
    staticGeom geometry = geoms[i];
    if (geometry.type == SPHERE){
      tmpDist = sphereIntersectionTest(geometry, thisRay, tmpIntersectPoint, tmpIntersectNormal);
    }else if (geometry.type == CUBE){
      tmpDist = boxIntersectionTest(   geometry, thisRay, tmpIntersectPoint, tmpIntersectNormal);
    }//insert triangles here for meshes
    if (tmpDist != -1 && tmpDist < distToIntersect){ //hit is new closest
      materialID = geometry.materialid;//index of hit material
      distToIntersect = tmpDist;
      intersectNormal = tmpIntersectNormal;
      intersectPoint  = tmpIntersectPoint;
    }
  }
  return distToIntersect;
}

__host__ __device__ glm::vec3 directLightContribution(material m, staticGeom* geoms, int numberOfGeoms, staticGeom* lights, int numberOfLights, 
	material* materials, glm::vec3 normal, glm::vec3 inDirection, glm::vec3 intersectionPoint, float rnd1, float rnd2, float& solidAngle){
  /////////////////////////////////////////////////
  //TODO: Update to support multiple light sources
  //  - Currently assumes all lights are spheres
  ////////////////////////////////////////////////
/*
  if(m.type == 1){
	  solidAngle = 0.0f;
	  return glm::vec3(0);

  }
*/
  
  //Get random point on light
  glm::vec3 lightPOS = getLightPos(lights, rnd1, rnd2); 
  float dist = glm::distance(lightPOS, intersectionPoint);
  
  //make ray
  ray thisRay;
  thisRay.origin = intersectionPoint;
  thisRay.direction = glm::normalize(lightPOS - thisRay.origin);
  //intersection checks
  float distToIntersect = dist; //distance to light intersection
  int materialID;                            //updated in intersectionTest
  glm::vec3 intersectPoint, intersectNormal; //updated in intersectionTest
  distToIntersect = intersectionTest(geoms, numberOfGeoms, materialID, thisRay, distToIntersect, intersectNormal, intersectPoint);
  
  glm::vec3 dirColor;
  if(distToIntersect < dist && materialID != 9){//in shadow
    dirColor = glm::vec3(0,0,0);
  }else{
    material lightMaterial = materials[lights[0].materialid];
    glm::vec3 lightColor = lightMaterial.color * lightMaterial.emittance;
    ///////////////////////////////
    //MODIFY THIS FOR OTHER BSDFS
    //////////////////////////////
    dirColor = getColorFromBSDF(inDirection, thisRay.direction, normal, lightColor, m);
  }
  //calculate solid angle
  solidAngle = getSolidAngle(lights[0], intersectionPoint, normal);
  return dirColor;
}

//Build Eye Path
__global__ void buildEyePath(glm::vec2 resolution, float time, cameraData cam, int maxDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, 
                            rayState* rayList, int currDepth, Path* eyePaths, staticGeom* lights, int numberOfLights){
  // index into array is based off pixel position
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
    rayState rs = rayList[index];
    if(rs.isValid == 0){
      eyePaths[index].vert[currDepth].isValid = 0;
      return;
    }
    //clear vertices
    vertex v;
    v.isValid  = 1;
    v.hitLight = 0;
    
    
    //random number generator
    thrust::default_random_engine rng(hash(index * (time + currDepth)));
    thrust::uniform_real_distribution<float> u01(0,1);
    
    //get variables
    ray thisRay     = rs.RAY;
    glm::vec3 COLOR = rs.color;
    v.inDirection = thisRay.direction;
    
    //intersection checks:
    float distToIntersect = FLT_MAX;
    int materialID = 0;                        //updated in intersectionTest
    glm::vec3 intersectPoint, intersectNormal; //updated in intersectionTest
    distToIntersect = intersectionTest(geoms, numberOfGeoms, materialID, thisRay, distToIntersect, intersectNormal, intersectPoint);
    material mat = materials[materialID];
    
    float solidAngle;
    glm::vec3 directLight;
    
    if(distToIntersect == FLT_MAX){//miss
      v.isValid = 0;
      eyePaths[index].vert[currDepth] = v; //update invalid vertex!
      rayList[index].isValid = 0;
      return;
    }else if(mat.type == 9){  //is this a light source?
      solidAngle = 1.0;//TWO_PI;//noDirectLight
      directLight = (mat.color * mat.emittance);
      COLOR = COLOR * directLight;
      v.hitLight = 1;
      v.isValid = 1;
      rayList[index].isValid = 0;
    }else{
	  solidAngle = 0.0;
      directLight = directLightContribution(mat, geoms, numberOfGeoms, lights, numberOfLights, materials, intersectNormal, thisRay.direction, intersectPoint, (float) u01(rng) ,(float) u01(rng), solidAngle); //updates solidAngle as side effect
    }


    //save intersection point to eyePath
    v.position = intersectPoint;
    
    //update variables
    float pdfWeight = 0;
    calculateBSDF(thisRay, intersectPoint, intersectNormal, COLOR, mat, (float) u01(rng) ,(float) u01(rng), pdfWeight, lights); 

    //update rayList
    rs.RAY   = thisRay;
    rs.color = COLOR;
    rayList[index] = rs;
    
    //save color to eyePath
    v.colorAcc = COLOR; //Saves color at each vertex although i think we only need the last one???
    v.normal = intersectNormal;
    v.outDirection = thisRay.direction; 
    v.mat = mat;

    //update PathProbability
    if(currDepth == 0){
      v.pathProbability = pdfWeight;
    }else{
      v.pathProbability = eyePaths[index].vert[currDepth - 1].pathProbability * pdfWeight; //Update Path Weight
    }
    v.directLight = directLight;
    v.solidAngle = solidAngle;//convertSolidAngle(solidAngle, distToIntersect, glm::dot(intersectNormal, thisRay.direction)) * solidAngle;
    v.pdfWeight = pdfWeight;  //probability of this bounce only
    eyePaths[index].vert[currDepth] = v;
  }
}

//Build Light Path
__global__ void buildLightPath(glm::vec2 resolution, float time, cameraData cam, int maxDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials,
                            rayState* rayList, int currDepth, Path* lightPaths, staticGeom* lights, int numberOfLights){
  // index into array is based off pixel position
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
    rayState rs = rayList[index];
    if(rs.isValid == 0){
      lightPaths[index].vert[currDepth].isValid = 0;
      return;
    }
    //clear vertices
    vertex v;
    v.isValid  = 1;
    v.hitLight = 0;


    //random number generator
    thrust::default_random_engine rng(hash(index * (time + currDepth)));
    thrust::uniform_real_distribution<float> u01(0,1);

    //get variables
    ray thisRay     = rs.RAY;
    glm::vec3 COLOR = rs.color;
    v.outDirection =  - thisRay.direction; //in light paths we're going backward

    //intersection checks:
    float distToIntersect = FLT_MAX;
    int materialID = 0;                        //updated in intersectionTest
    glm::vec3 intersectPoint, intersectNormal; //updated in intersectionTest
    distToIntersect = intersectionTest(geoms, numberOfGeoms, materialID, thisRay, distToIntersect, intersectNormal, intersectPoint);
    material mat = materials[materialID];

    float solidAngle;
    glm::vec3 directLight;

    if(distToIntersect == FLT_MAX){//miss
      v.isValid = 0;
      lightPaths[index].vert[currDepth] = v; //update invalid vertex!
      rayList[index].isValid = 0;
      return;
    }else if(mat.type == 9){  //is this a light source?
      solidAngle = 1.0f;//TWO_PI;//noDirectLight
      directLight = (mat.color * mat.emittance);
      COLOR = COLOR * directLight;
      v.hitLight = 1;
      v.isValid = 0;
      rayList[index].isValid = 0; //I could probably let it continue...
    }else{
	  solidAngle = 0.0;
      directLight = directLightContribution(mat, geoms, numberOfGeoms, lights, numberOfLights, materials, intersectNormal, thisRay.direction, intersectPoint, (float) u01(rng) ,(float) u01(rng), solidAngle); //updates solidAngle as side effect
    }


    //save intersection point to lightPath
    v.position = intersectPoint;

    //update variables
    float pdfWeight = 0;
    calculateBSDF(thisRay, intersectPoint, intersectNormal, COLOR, mat, (float) u01(rng) ,(float) u01(rng), pdfWeight, lights);

    //update rayList
    rs.RAY   = thisRay;
    rs.color = COLOR;
    rayList[index] = rs;

    //save color to eyePath
    v.colorAcc = COLOR; //Saves color at each vertex although i think we only need the last one???
    v.normal = intersectNormal;
    v.inDirection = - thisRay.direction; // we're tracing backward
    v.mat = mat;

    //update PathProbability
    //start at depth 1 not 0;
    v.pathProbability = lightPaths[index].vert[currDepth - 1].pathProbability * pdfWeight; //Update Path Weight

    v.directLight = directLight;
    v.solidAngle = solidAngle;//convertSolidAngle(solidAngle, distToIntersect, glm::dot(intersectNormal, thisRay.direction)) * solidAngle;
    v.pdfWeight = pdfWeight;  //probability of this bounce only
    lightPaths[index].vert[currDepth] = v;
  }
}


__global__ void connectPaths(glm::vec2 resolution, glm::vec3* colors, float* imageWeights, staticGeom* geoms, int numberOfGeoms, int traceDepth, Path* eyePaths, Path* lightPaths){
  // index into array is based off pixel position
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if((x<=resolution.x && y<=resolution.y)){
    
    //updates all eye paths that hit a light source
    for (int lightIDX = 0; lightIDX < 10; lightIDX++){
      for (int idx = 0; idx < traceDepth; idx++){//traceDepth - 4; // First bounce of light
        for(int eyeVert = 0; eyeVert < traceDepth; eyeVert++){
          if (eyePaths[index].vert[eyeVert].isValid != 0 && lightPaths[lightIDX].vert[idx].isValid != 0){
            ray thisRay; 
            thisRay.origin = eyePaths[index].vert[eyeVert].position; 
            thisRay.direction = glm::normalize(lightPaths[lightIDX].vert[idx].position - eyePaths[index].vert[eyeVert].position);
            //check intersection of this ray with scene
            float dist = glm::distance(lightPaths[lightIDX].vert[idx].position, thisRay.origin);
            float distToIntersect = dist;
            int materialID = 0;                        //updated in intersectionTest
            glm::vec3 intersectPoint, intersectNormal; //updated in intersectionTest
            distToIntersect = intersectionTest(geoms, numberOfGeoms, materialID, thisRay, distToIntersect, intersectNormal, intersectPoint);
            
            if(distToIntersect == dist){ //no intersection, we can add color
            	 //change weight calculation when we add other materials
            	float weight = imageWeights[index];
            	float denom  = weight + 1.0f;
            	glm::vec3 pathColor = eyePaths[index].vert[eyeVert].colorAcc  * lightPaths[lightIDX].vert[idx].colorAcc;
            	colors[index] = colors[index] * (weight/denom) + pathColor * (1.0f /denom);
            	imageWeights[index] = denom;
          //return;
            }
          }
        }
      }
    }
  }
}
//NO DIRECT LIGHTING CONTRIBUTION
__global__ void RenderColor(glm::vec2 resolution, glm::vec3* colors, float* imageWeights, int traceDepth, Path* eyePaths) {
  // index into array is based off pixel position
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //integrate light contribution Back to Front.
  if(x<=resolution.x && y<=resolution.y){
    for(int vert = traceDepth - 1; vert >= 0; vert--){
      if(eyePaths[index].vert[vert].isValid == 1 && eyePaths[index].vert[vert].hitLight == 1){
        float weight = imageWeights[index];
        float pdfWeight = eyePaths[index].vert[vert].pathProbability;
        float denom  = weight + pdfWeight;
        colors[index] = colors[index] * (weight/denom) + eyePaths[index].vert[vert].colorAcc * (pdfWeight /denom);
        imageWeights[index] = denom;
        return;
      }
    }
  }
}

//Only render direct lighting on first bounce
__global__ void RenderDirectLight(glm::vec2 resolution, glm::vec3* colors, float* imageWeights, int traceDepth, Path* eyePaths) {
  // index into array is based off pixel position
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //integrate light contribution Back to Front.
  if(x<=resolution.x && y<=resolution.y){
    int vert = 0;
    if(eyePaths[index].vert[vert].isValid){
        float weight = imageWeights[index];
        float solidAngle = eyePaths[index].vert[vert].solidAngle;
        float denom  = weight + solidAngle;
        colors[index] = colors[index] * (weight/denom) + eyePaths[index].vert[vert].directLight * (solidAngle /denom);
        imageWeights[index] = denom;
        return;
      }
  }
}


__global__ void MISRenderColor(glm::vec2 resolution, glm::vec3* colors, float* imageWeights, int traceDepth, Path* eyePaths, float time, staticGeom* geoms, 
	int numberOfGeoms, material* materials) {
  // index into array is based off pixel position
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //integrate light contribution Back to Front.
  if(x<=resolution.x && y<=resolution.y){
    glm::vec3 inDirection;
    glm::vec3 outDirection;
    glm::vec3 normal;
    ray thisRay;
    glm::vec3 BSDFcolor = glm::vec3(0);
    glm::vec3 inColor = glm::vec3(0);
    float solidAngle;
    float pdfWeight;
    glm::vec3 directLight;
    
    int validRay = 0;
    float totalPDFWeight = 0.0f;
    vertex v;
    int max = traceDepth - 1;
    for(int vert = max ; vert >= 0; vert--){
      v = eyePaths[index].vert[vert];
      if(v.isValid == 1){
        validRay = 1;
        material mat = v.mat;
        if(v.hitLight == 1){
          //This vertex is on a light
          inColor = mat.color * mat.emittance;
          totalPDFWeight = 1.0f;
        }else{
          totalPDFWeight *= v.pdfWeight;
          //update BSDF color
          inDirection  = v.inDirection;
          outDirection = v.outDirection;
          normal       = v.normal;
          pdfWeight    = v.pdfWeight;
          BSDFcolor    = getColorFromBSDF(inDirection, outDirection, normal, inColor, mat);
         inColor = BSDFcolor;
          /*
          //update incoming color
          solidAngle  = v.solidAngle;
          directLight = v.directLight;
          
          //power heuristic
          pdfWeight  *= pdfWeight;
          solidAngle *= solidAngle;
          
          // balance heuristic to update incolor
          float denom = solidAngle + pdfWeight;
          //inColor     = (solidAngle/denom) * directLight + (pdfWeight/denom) * BSDFcolor;
          inColor     = directLight + BSDFcolor;
          */
        }
      }
    }
    if(validRay == 1){
      //Update Pixel Color
    	/*
      float weight = imageWeights[index];
      float denom  = weight + totalPDFWeight;
      colors[index] = colors[index] * (weight/denom) + inColor * (totalPDFWeight/denom);
      imageWeights[index] = denom;
      */
    	v = eyePaths[index].vert[0];
    	pdfWeight = totalPDFWeight;
    	 solidAngle = v.solidAngle;

		 //pdfWeight *= pdfWeight;
		 //solidAngle *= solidAngle;

		 float denom = pdfWeight + solidAngle;
		 inColor = inColor * (pdfWeight/denom) + v.directLight * (solidAngle/denom);
		 float weight = imageWeights[index];
		 denom  = weight + 1.0f;
		 //debug:
		 //inColor = glm::vec3(pdfWeight);

		 colors[index] = colors[index] * (weight/denom) + inColor * (1.0f/denom);
		 imageWeights[index] = denom;
    }
  }
}

__global__ void BiDirRenderColor(glm::vec2 resolution, glm::vec3* colors, float* imageWeights, int traceDepth, Path* eyePaths, float time, staticGeom* geoms,
	int numberOfGeoms, material* materials, Path* lightPaths) {
  // index into array is based off pixel position

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  //integrate light contribution Back to Front.
  if(x<=resolution.x && y<=resolution.y){
	glm::vec3 inDirection;
    glm::vec3 outDirection;
    glm::vec3 normal;
    glm::vec3 averageBSDF;
    float sumPDF = 0.0f;
    float numPDFs = 0.0f;
    ray thisRay;
    glm::vec3 BSDFcolor = glm::vec3(0);
    glm::vec3 inColor;
    float solidAngle;
    float pdfWeight;
    glm::vec3 directLight;
    int validRay = 0;
    float totalPDFWeight = 1.0f;
    material mat;
    vertex v;

    //for each potential light path calculate
    // - Incoming Light (flux retained from light)
    // - PDFWeight (accumulated)
    // - position of final vertex
    for(int LP = 0; LP <= 4; LP++){
		//int LP = 2;
    	vertex lv = lightPaths[LP].vert[0];
		glm::vec3 lightColor = lv.mat.color * lv.mat.emittance;
		float lightPDF = 1.0f;
		for(int li = 1; li <= traceDepth; li ++){
			//connect light vertex to each eye vertex
			for( int max = traceDepth - 1; max >=0; max--){
				totalPDFWeight = lightPDF;
				inColor = lightColor;
				validRay = 0;
				v = eyePaths[index].vert[max];
				//check if can connect.  if no break
				 if(v.isValid == 1){
				//intersection checks:
					float dist = glm::distance(lv.position, v.position);
					float distToIntersect = dist;
					thisRay.direction = glm::normalize(lv.position - v.position);
					thisRay.origin = v.position;

					int materialID = 0;                        //updated in intersectionTest
					glm::vec3 intersectPoint, intersectNormal; //updated in intersectionTest
					distToIntersect = intersectionTest(geoms, numberOfGeoms, materialID, thisRay, distToIntersect, intersectNormal, intersectPoint);
					if(distToIntersect < dist){//hit something
						material m = materials[materialID];
						if(m.type != 9){
							continue;
						}
					}
					//update inColor
					inColor = getColorFromBSDF(v.inDirection, thisRay.direction , v.normal, inColor, v.mat);
					//update Probability
					pdfWeight = PDF(v.inDirection, thisRay.direction, v.normal, mat);
					totalPDFWeight *= pdfWeight;
					for(int vert = max - 1 ; vert >= 0; vert--){
					  v = eyePaths[index].vert[vert];
					  if(v.isValid == 1){
						validRay = 1;
						material mat = v.mat;
						if(v.hitLight == 1){
							inColor = mat.color * mat.emittance;
							totalPDFWeight = 1.0f;//*= PDF(v.inDirection, v.outDirection, v.normal, v.mat);
						}else{
						  totalPDFWeight *= v.pdfWeight;
						  //update BSDF color
						  BSDFcolor    = getColorFromBSDF(v.inDirection, v.outDirection, v.normal, inColor, v.mat);
						  //inColor     = v.directLight + BSDFcolor;
						  inColor = BSDFcolor;
						  /*
						   BSDFcolor    = getColorFromBSDF(v.inDirection, v.outDirection, v.normal, inColor, v.mat);

						   //power heuristic
						   pdfWeight  = v.pdfWeight * v.pdfWeight;
						   solidAngle = v.solidAngle * v.solidAngle;

						   // balance heuristic to update incolor
						   float denom = solidAngle + pdfWeight;
						   inColor     = (solidAngle/denom) * v.directLight + (pdfWeight/denom) * BSDFcolor;
						   //inColor     = directLight + BSDFcolor;
							*/
						  //inColor      = getColorFromBSDF(v.inDirection, v.outDirection, v.normal, inColor, v.mat);

						}
					  }
					}
					if(validRay == 1 && totalPDFWeight > 0){
/*
						//Update Pixel Color
						float weight = imageWeights[index];
						float denom  = weight + totalPDFWeight;
						//float MISDenom = eyePaths[index].vert[0].pdfWeight + eyePaths[index].vert[0].solidAngle;
						colors[index] = colors[index] * (weight/denom) + inColor * (totalPDFWeight/denom);
						imageWeights[index] = denom;
*/
						float weight = sumPDF;
						float denom = sumPDF + totalPDFWeight;
						averageBSDF = averageBSDF * (weight/denom) + inColor * (totalPDFWeight/denom);
						sumPDF = denom;
						numPDFs += 1.0f;
					}
				 }
				}
			// Update light vertex
			if(li < traceDepth){
				lv = lightPaths[LP].vert[li];
				if(lv.isValid == 0){
					break; //unless we want to do the second pass on eye paths below
				}
				lightColor = getColorFromBSDF(lv.inDirection, lv.outDirection, lv.normal, lightColor, lv.mat);
				lightPDF *= lv.pdfWeight;
			}
		}
    }
    //MIS HEURISTIC WITH DIRECT LIGHT!
	v = eyePaths[index].vert[0];//direct light first bounce
	//power heuristic
	if(numPDFs > 0.0){
		pdfWeight  = sumPDF/numPDFs;
	}else{
		pdfWeight = 0.0;
	}
	if(v.isValid == 1){

		 solidAngle = v.solidAngle;

		 pdfWeight *= pdfWeight;
		 solidAngle *= solidAngle;

		 float denom = pdfWeight + solidAngle;
		 inColor = averageBSDF * (pdfWeight/denom) + v.directLight * (solidAngle/denom);
		 float weight = imageWeights[index];
		 denom  = weight + 1.0f;
		 //debug:
		 //inColor = glm::vec3(pdfWeight);

		 colors[index] = colors[index] * (weight/denom) + inColor * (1.0f/denom);
		 imageWeights[index] = denom;


	}else if (pdfWeight > 0.0f){
		float weight = imageWeights[index];
		float denom  = weight + 1.0f;
		colors[index] = colors[index] * (weight/denom) + averageBSDF * (1.0f/denom);
		imageWeights[index] = denom;
	}
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


// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms, geom* lights, int numberOfLights, int renderType){
  
  int traceDepth = 4; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  // send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage,           (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // allocate eye path per pixel
  Path* eyePaths = NULL;
  cudaMalloc((void**)&eyePaths,           (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(Path));

  // allocate Light paths
	Path* lightPaths = NULL;
	//cudaMalloc((void**)&lightPaths,         (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(Path));
	cudaMalloc((void**)&lightPaths,         10 * sizeof(Path));

  // Allocate per-pixel accumulated weight (probabilites of valid light paths)
  float* imageWeights = NULL;
  cudaMalloc((void**)&imageWeights,                  (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(float));
  cudaMemcpy( imageWeights, renderCam->imageWeights, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(float), cudaMemcpyHostToDevice);
  
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
  
  
  // package Lights and sent to GPU
  staticGeom* lightList = new staticGeom[numberOfLights];
  for(int i=0; i<numberOfLights; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = lights[i].type;
    newStaticGeom.materialid = lights[i].materialid;
    newStaticGeom.translation = lights[i].translations[frame];
    newStaticGeom.rotation = lights[i].rotations[frame];
    newStaticGeom.scale = lights[i].scales[frame];
    newStaticGeom.transform = lights[i].transforms[frame];
    newStaticGeom.inverseTransform = lights[i].inverseTransforms[frame];
    lightList[i] = newStaticGeom;
  }
  
  staticGeom* cudalights = NULL;
  cudaMalloc((void**)&cudalights,    numberOfLights * sizeof(staticGeom));
  cudaMemcpy( cudalights, lightList, numberOfLights * sizeof(staticGeom), cudaMemcpyHostToDevice);
  
  
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


  //allocate light rays 
  rayState* lightrayList = NULL;
  //cudaMalloc((void**)&lightrayList, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(rayState));
  cudaMalloc((void**)&lightrayList, 10 * sizeof(rayState));
  
 // kernel launches
  //Get initial rays
  initializeRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, rayList);

  //Initialize light subpaths
  int numLightpaths = 10;

  initializeLightPaths<<<1, numLightpaths>>>((float)iterations, cam, lightrayList, numLightpaths,  cudalights, numberOfLights, lightPaths,  materialList);
  

  //build eye path
  for(int i = 0; i < traceDepth; i++){
    //do one step
    buildEyePath<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, materialList, numberOfMaterials, rayList, i, eyePaths, cudalights, numberOfLights);
  }

/*
  // error checking
  Path* localPaths = new Path[(int)renderCam->resolution.x * (int)renderCam->resolution.y];
  cudaMemcpy( localPaths, eyePaths, (int)renderCam->resolution.x * (int)renderCam->resolution.y * sizeof(Path), cudaMemcpyDeviceToHost);
  for(int i = 0; i < (int)renderCam->resolution.x * (int)renderCam->resolution.y; i++){
	  vertex v = localPaths[i].vert[1];
	  if(v.isValid == 1){
		  if(v.solidAngle < 0.0 || v.solidAngle > 1.0){
			  std::cout << "V" << i << " solid angle: " << v.solidAngle << std::endl;
			  std::cout << "V" << i << " pdfWeight: " << v.pdfWeight << std::endl;
		  }else if (v.pdfWeight < 0.0 || v.pdfWeight > 1.0){
			  //std::cout << "V" << i << " solid angle: " << v.solidAngle << std::endl;
			  std::cout << "V" << i << " pdfWeight: " << v.pdfWeight << std::endl;
		  }
	  }
  }
  exit(0);
*/

  //build light path
    for(int i = 1; i < traceDepth; i++){
      //do one step
      buildLightPath<<<1, numLightpaths>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, materialList, numberOfMaterials, lightrayList, i, lightPaths, cudalights, numberOfLights);
    }
/*
   //buildLightPath
  for(int i = 0; i < traceDepth; i++){
    //do one step
    buildEyePath<<<1, numLightpaths>>>(glm::vec2(10,1), (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, materialList, numberOfMaterials, lightrayList, i, lightPaths);
  }
*/


/*  
  //connect paths and render to screen
  connectPaths<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, imageWeights, cudageoms, numberOfGeoms, traceDepth, eyePaths, lightPaths);
*/
if(renderType == 0){//classic PathTracer
  RenderColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, imageWeights, traceDepth, eyePaths);
}else if(renderType == 1){ //Direct Lighting Only
  RenderDirectLight<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, imageWeights, traceDepth, eyePaths);
}else if(renderType == 2){//Multiple Importance Sampling
  MISRenderColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, imageWeights, traceDepth, eyePaths, (float)iterations, cudageoms, numberOfGeoms, materialList);
}else{
	BiDirRenderColor<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, imageWeights, traceDepth, eyePaths, (float)iterations, cudageoms, numberOfGeoms, materialList, lightPaths);
}
  //update visual
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  // retrieve image from GPU
  cudaMemcpy( renderCam->image,        cudaimage,    (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  //retrieve weights from GPU
  cudaMemcpy( renderCam->imageWeights, imageWeights, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(float), cudaMemcpyDeviceToHost);
/*
  //debug
  for(int i = 0; i < (int)renderCam->resolution.x * (int)renderCam->resolution.y; i++){
	  float weight = renderCam->imageWeights[i];
	  if(weight == 0.0 || weight == 1.0){
		  std::cout << "pixel: " << i << " weight: " << weight << std::endl;
	  }
  }
  exit(0);
*/

  // free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudalights ); //added
  cudaFree(materialList); //added
  cudaFree(rayList);      //added
  cudaFree(lightrayList); //VCM added
  cudaFree(eyePaths);     //added
  cudaFree(lightPaths);     //added
  cudaFree(imageWeights); //added
  delete geomList;
  delete lightList;//ADDED

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}

