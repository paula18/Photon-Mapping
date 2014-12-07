// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H


#include "intersections.h"

/////////////////////////////////
// Forward Declarations
/////////////////////////////////
__host__ __device__ float PDFSpecular(glm::vec3, glm::vec3, glm::vec3, float);
__host__ __device__ float PDFDiffuse(glm::vec3 normal, glm::vec3 direction);
__host__ __device__ glm::vec3 randomPointOnLight(staticGeom light, float rnd1, float rnd2);

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
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR)
{
	Fresnel fresnel;

	float n12 = incidentIOR / transmittedIOR; 

	float cosThetaI = -1.0f * glm::dot(incident, normal); 
	float sin2ThetaT = n12 * n12 * (1 - cosThetaI * cosThetaI); 
	float cosThetaT = sqrt(1 - sin2ThetaT);

	float a = (incidentIOR - transmittedIOR) / (incidentIOR + transmittedIOR);
	float R0 = a * a;

	float b5 = (1 - cosThetaI) * (1 - cosThetaI) * (1 - cosThetaI) * (1 - cosThetaI)
		  * (1 - cosThetaI);

	float c5 = (1 - cosThetaT) * (1 - cosThetaT) * (1 - cosThetaT) * (1 - cosThetaT) 
		  * (1 - cosThetaT); 

	float Rschlick; 
	float Tschlick; 


	//Schlick's approx
	if ( incidentIOR <= transmittedIOR)
		  Rschlick = R0 + (1 - R0) * b5; 
	
	else if ( (incidentIOR > transmittedIOR) && (sin2ThetaT <= 1.0f) )
		  Rschlick = R0 + (1 - R0) * c5; 
	  
	else if ( (incidentIOR > transmittedIOR) && (sin2ThetaT > 1.0f) )
		  Rschlick = 1; 

	Tschlick = 1 - Rschlick; 

	
	fresnel.reflectionCoefficient = Rschlick;
	fresnel.transmissionCoefficient = Tschlick;
	
	return fresnel;
}

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



__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2, glm::vec3 center) {
/*
	float alpha = xi2 * TWO_PI; 
	float phi = glm::acos(2 * xi1 - 1);
	float x = center.x + (sin(phi) * cos(alpha));
	float y = center.y + (sin(phi) * sin(alpha));
	float z = center.z + (cos(phi));

	return glm::normalize(glm::vec3(x, y, z));
*/
	float theta = 2.0f * PI * xi1;
	float phi = acos(2.0f * xi2 - 1);
	float radius = 1.0f;
	glm::vec3 localPosition = glm::vec3( radius * sin(theta) * cos(phi), radius * sin(theta) * sin(phi), radius * cos(phi));

	return localPosition + center;
}


//materialType  0 = diffuse
__host__ __device__ glm::vec3 getColorFromBSDF(glm::vec3 inDirection, glm::vec3 toLight, glm::vec3 normal, glm::vec3 lightColor, material mat){
  //First decide material type
	if(mat.type == 0)
	{
		float cos = max(glm::dot(toLight, normal),0.0);
		glm::vec3 color = lightColor * cos * mat.color;
		return color;
	}
	else if (mat.type == 1)
	{
		float specPDF = PDFSpecular(inDirection, toLight, normal, 100);
		glm::vec3 color = lightColor * mat.specularColor * specPDF;
		// add diffuse component (screws up direct lighting)
		//float cos = max(glm::dot(toLight, normal),0.0);
		//color += lightColor * cos * mat.color;
		return color;
	}
	return lightColor;//should be unreachable.  
}

__host__ __device__ glm::vec3 getLightPos(staticGeom *lights, float rnd1, float rnd2)
{
	// choose light at random
	staticGeom light = lights[0];
	return randomPointOnLight(light, rnd1, rnd2);
}

__host__ __device__ glm::vec3 randomPointOnLight(staticGeom light, float rnd1, float rnd2)
{

	glm::vec3 lightPos = getRandomDirectionInSphere(rnd1, rnd2, glm::vec3(0,0,0)); // random point on unit sphere
	lightPos = multiplyMV(light.transform, glm::vec4(lightPos,1.0f));         // translate to point on light
	return lightPos;
}

///////////////////////////////////////////////
// BSDFs CALCULATIONS
//////////////////////////////////////////////

///////////////////////////////////////////////
// DIFFUSE
//////////////////////////////////////////////

//Updates color and ray
__host__ __device__ int calculateDiffuseDirection(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                      glm::vec3& color, material mat, float seed1, float seed2)
{
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

//Returns PDF for diffuse materials
__host__ __device__ float PDFDiffuse(glm::vec3 normal, glm::vec3 direction)
{
	return glm::clamp( glm::dot( normal, direction ), 0.0f, 1.0f )*INV_PI;
}

//Stores color, PDFWeight and updates ray. 
__host__ __device__ void calculateDiffuseBSDF(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2, float& PDFWeight)
{
	calculateDiffuseDirection(thisRay, intersect, normal, color, mat, seed1, seed2);
	PDFWeight = PDFDiffuse(normal, thisRay.direction);
}

///////////////////////////////////////////////
// PERFECT SPECULAR
//////////////////////////////////////////////
__host__ __device__ glm::vec3 generateDir(float seed1, float seed2, float shininess){
	
	float phi = 2*PI*seed1; 
	float theta = acos(pow(seed2, (1.0f/(shininess+1.0f)))); 

	float x = cos(phi)*sin(theta); 
	float y = sin(phi)*sin(theta); 
	float z = cos(theta); 

	return glm::normalize(glm::vec3(x, y, z)); 
}

__host__ __device__ glm::vec3 localToWorld( glm::vec3 localDir, glm::vec3 normal )
{
    glm::vec3 binormal = glm::normalize( ( abs(normal.x) > abs(normal.z) )?glm::vec3( -normal.y, normal.x, 0.0 ):glm::vec3( 0.0, -normal.z, normal.y ) );
	glm::vec3 tangent = glm::cross( binormal, normal );
    
	return localDir.x*tangent + localDir.y*binormal + localDir.z*normal;
}
__host__ __device__ int calculateReflectiveDirection(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2, float shininess)
{
	ray newRay;
	 //Perfect reflective
    newRay.direction = glm::reflect(thisRay.direction, normal);
    newRay.direction = glm::normalize(newRay.direction);
	//newRay.direction = localToWorld(generateDir(seed1, seed2, shininess), newRay.direction);
    newRay.origin = intersect + .001f * newRay.direction;//nudge in direction
	//Update COLOR
	color = color * mat.specularColor; 
    thisRay = newRay;
    return 1;
}


__host__ __device__ float PDFSpecular(glm::vec3 viewDir, glm::vec3 lightDir, glm::vec3 normal, float shininess)
{
	glm::vec3 R = glm::reflect(viewDir, normal); 
	float d = glm::dot(R, lightDir); 
	//return max(0.0, pow(d, shininess))*(shininess+1)*max(0.0f,min(1.0, sin(acos(d))))/TWO_PI;
	//return glm::clamp( (float)(max(0.0f, pow(d, shininess))*(shininess+1.0f)/TWO_PI), 0.0f, 1.0f);
	return 1;
}

__host__ __device__ void calculateSpecularBSDF(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2, staticGeom * lights, float &PDFWeight,
									   float shininess)
{
	ray orig = thisRay;
	calculateReflectiveDirection(thisRay, intersect, normal, color, mat, seed1, seed2, shininess);
	//glm::vec3 lightPos = getLightPos(lights, seed1, seed2);   
	//glm::vec3 lightDir = lightPos - intersect; 
	PDFWeight = PDFSpecular(orig.direction, thisRay.direction, normal, shininess);

}

__host__ __device__ float PDF(glm::vec3 viewDir, glm::vec3 lightDir, glm::vec3 normal, material mat){
	if(mat.type == 0){ //|| mat.type == 9){//diffuse or light
		return PDFDiffuse(normal,lightDir);
	}else if(mat.type == 1){
		return PDFSpecular(viewDir, lightDir, normal, 100);
	}
	return 0.0f;
}


/*
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
*/

///////////////////////////////////////////////
// GENERAL BSDF CALCULATION 
//////////////////////////////////////////////

__host__ __device__ int calculateBSDF(ray& thisRay, glm::vec3 intersect, glm::vec3 normal,
                                       glm::vec3& color, material mat, float seed1, float seed2, float& PDFWeight, staticGeom *lights)
{
  // This updates the ray directions and color
	float pdf	;					
	int materialType;
  
  //Diffuse

	if(mat.type == 0 || mat.type == 9)
	{
		calculateDiffuseBSDF(thisRay, intersect, normal, color, mat, seed1, seed2, PDFWeight); 
		return materialType;
	}
	//Perfect reflection
	else if (mat.type == 1)
	{
		float shininess = 100;
		calculateSpecularBSDF(thisRay, intersect, normal, color, mat, seed1, seed2, lights, PDFWeight, shininess);
		return materialType; 
	}
	/*else if (mat.type == 2){
		PDFWeight = 1.0f; 
		return materialType; 
	}*/
};

#endif
