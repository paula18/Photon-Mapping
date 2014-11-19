// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"

struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient;
};

// Forward declaration
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident) {
  //nothing fancy here
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection) {
  Fresnel fresnel;

  fresnel.reflectionCoefficient = 1;
  fresnel.transmissionCoefficient = 0;
  return fresnel;
}

// LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    // Crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    // Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}


// Now that you know how cosine weighted direction generation works, try implementing 
// non-cosine (uniform) weighted random direction generation.
// This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {
  float randomSeed = xi1 * xi2;
  //from method found at http://mathworld.wolfram.com/SpherePointPicking.html
  thrust::default_random_engine rng(hash(randomSeed));
  thrust::uniform_real_distribution<float> u01(-1,1);
  //thrust::uniform_real_distribution<float> u02(-1,1);
  float x_1 = 2;
  float x_2 = 2;
  float x_1_squared = 4;
  float x_2_squared = 4;
  while((x_1_squared + x_2_squared) >= 1){ //reject where sum of squares >= 1
    x_1 = (float)u01(rng);
    x_2 = (float)u01(rng);
    x_1_squared = x_1 * x_1;
    x_2_squared = x_2 * x_2;
  }
  float x, y, z;
  x = 2 * x_1 + sqrt(1 - x_1_squared - x_2_squared);
  y = 2 * x_2 + sqrt(1 - x_1_squared - x_2_squared);
  z = 1 - 2 * (x_1_squared + x_2_squared);
  
  return glm::vec3(x,y,z);
}

__host__ __device__ int calculateReflective(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2){
   ray newRay;
   //Perfect reflective
    newRay.direction = glm::reflect(thisRay.direction, normal);
    newRay.direction = glm::normalize(newRay.direction);
    newRay.origin = intersect + .001f * newRay.direction;//nudge in direction
    thisRay = newRay;
    return 1;
}

__host__ __device__ int calculateRefractive(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2){
//consulted Bram de Grave paper 2006 Reflections and Refractions in Ray Tracing for help with algorithm.
  ray newRay;
  float theta, phi;
  float indexOfRefraction = mat.indexOfRefraction;
  
  //refraction angle
  theta = glm::dot(normal, thisRay.direction);
  if (theta > 0){
    //flip normal, i'm inside the object
    normal = - normal;
  }else{
    //flip theta and invert IOR
    theta = -theta;
    indexOfRefraction = 1/indexOfRefraction;
  }
  phi = indexOfRefraction * indexOfRefraction * (1 - theta * theta);
  //is there total internal reflection?
  if (phi > 1){
    return calculateReflective(thisRay, intersect, normal, color, mat, seed1, seed2); //switch to reflection
  }
  float sinPhi = sqrt(1 - phi);
  newRay.direction = indexOfRefraction * thisRay.direction + (indexOfRefraction * theta - sinPhi) * normal;
  newRay.direction = glm::normalize(newRay.direction);
  newRay.origin = intersect + .001f * newRay.direction; //nudge in direction
  thisRay = newRay;
  return 2;
}

__host__ __device__ int calculateDiffuse(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2){
    ray newRay;
    //Diffuse
    newRay.direction = calculateRandomDirectionInHemisphere(normal, seed1, seed2);
    newRay.direction = glm::normalize(newRay.direction);
    newRay.origin = intersect + .001f * newRay.direction; //nudge in direction
    //get Cosine of new ray and normal
    float cos = glm::dot(newRay.direction, normal);
    //update COLOR
    color = color * mat.color * cos;
    thisRay = newRay;
    return 0;
}

// TODO (PARTIALLY OPTIONAL): IMPLEMENT THIS FUNCTION
		///////////////////////////////////
		//////////////////////////////////
		// TODO: IMPLEMENT THIS FUNCTION/
		////////////////////////////////
		///////////////////////////////
// Returns 0 if diffuse scatter, 1 if reflected, 2 if transmitted.
/*__host__ __device__ int calculateBSDF(ray& r, glm::vec3 intersect, glm::vec3 normal, glm::vec3 emittedColor,
                                       AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                       glm::vec3& color, glm::vec3& unabsorbedColor, material m){ */
__host__ __device__ int calculateBSDF(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2){
  if((seed1 + seed2) > 1){
    //check reflectance first
    if(seed2 < mat.hasReflective){ 
      return calculateReflective(thisRay, intersect, normal, color, mat, seed1, seed2);
    }else if (seed2 < mat.hasRefractive){
      return calculateRefractive(thisRay, intersect, normal, color, mat, seed1, seed2);
    }else{
      return calculateDiffuse(thisRay, intersect, normal, color, mat, seed1, seed2);
    }
  }else{
    //check refractive first
    if(seed1 < mat.hasRefractive){
      return calculateRefractive(thisRay, intersect, normal, color, mat, seed1, seed2);
    }else if (seed1 < mat.hasReflective){
      return calculateReflective(thisRay, intersect, normal, color, mat, seed1, seed2);
    }else{
      return calculateDiffuse(thisRay, intersect, normal, color, mat, seed1, seed2);
    }
  }
};

#endif
