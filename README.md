# Robotics-Based bronchoscopy
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

## How to Use This Code

### Download the Code
Clone or download this repository to your local machine.  
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/RikkeB2/Broncho-Project.git)  
cd Broncho-Project  

### Run the LEBO Model
To execute the LEBO model, run the following script:  
Example: python code/VS_LEBO.py  

### Visualize Log Files
To visualize outputs or log files, use the scripts available in the code/visualisation/ folder. These are helpful for analyzing and plotting results from model runs.  
Example:  python code\visualization\plot_pcd.py   

## Acknowledgements
This project builds on code provided by our co-supervisor as part of the BronchoBot project. We gratefully acknowledge their support and contribution. You can learn more about the BronchoBot project here: [https://portal.findresearcher.sdu.dk/da/projects/bronchobot]
