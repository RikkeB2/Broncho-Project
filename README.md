# Robotics-based bronchoscopy
Bachelorproject @ SDU, F25

Students:
+ Rikke Aalling Boysen (riboy22@student.sdu.dk),
+ Simone Ingvild Lebech (sileb18@student.sdu.dk),  
  
Supervior: Thiusius R. Savarimuthu (trs@mmmi.sdu.dk),  
Co-supervisor: Bruno Oliveira (broli@mmmi.sdu.dk)

## Problem Statement
While robotic bronchoscopy has advanced the field by offering greater precision and consistency than manual procedures, most systems remain dependent on preoperative CT imaging for path planning. 
This reliance limits adaptability during procedures and adds cost, time, and radiation exposure. This project investigates whether autonomous bronchoscopy can be achieved without CT data, using a 
real-time kinematic control strategy combined with monocular depth estimation to navigate through reconstructed airway geometries. Instead of relying on full SLAM, the system uses image-derived 
spatial information to continuously update its path and articulation in response to the environment. The key challenge is whether such a geometry-aware, CT-free approach can navigate the airways 
with sufficient accuracy and safety under controlled conditions. To guide and evaluate this investigation, the following hypotheses have been formulated:

### Hypothesis 1 - Autonomous Navigation Feasibility
A robotic bronchoscopy system using monocular depth and pose estimation for 3D lumen reconstruction, together with graph-based trajectory planning, will enable autonomous navigation through a realistic lung phantom. 
Success will be defined as autonomous traversal of $\geq80\%$ of a predefined central airway path, with $\leq1$ collision per trajectory.

### Hypothesis 2 – Realistic, Stable Motion via Control
Incorporating clinically inspired joint velocity and acceleration limits into the control strategy will result in smoother and more biologically plausible articulation compared to an unconstrained baseline. 
This constraint-aware approach is expected to reduce joint instability and improve motion continuity, particularly in anatomically complex regions. Success will be defined as maintaining an average Euclidean 
deviation from the centreline under 10 mm.
