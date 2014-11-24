Bidirectional Path Tracer
==============
![alpha_presentation](https://raw.githubusercontent.com/paula18/Photon-Mapping/master/renders/bidir_alpha.png)

==============
ALPHA Presentation
==============
For the Alpha step of the project we have modified our Pathtracer to trace in both directions.  We send rays from the eye into the scene, and we also send rays from a light into the scene.  

At each step of the eyePath and lightPath we save the position and accumulated light values.  We then have a separate connect kernel which connects the eye Paths to the Light paths.  This works in the following way:

==============
Connect Paths
==============
For each vertex in the eye Path we shoot a ray to a vertex in the Light Path.  If there is no geometry blocking them we connect the rays at this point.  

We repeat this for every eye Vertex/Light Vertex combination.  While this slows down the running time of each iteration significantly, each iteration is much cleaner because it has gathered exponentially more paths per pixel.

* The caveat of this is that while it works for diffuse surfaces, it will not work for reflective surfaces. 
* The next step of our project will be to implement Multiple Importance Sampling.  We will do this for both the unidirectional pathtracer and the bidirectional pathtracer so that we can compare the performance of each.