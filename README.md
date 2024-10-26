# AIS-Trajectory-Matching-From-Radar-Points
This GitHub repository contains the implementation code of the algorithm I proposed in my paper 'A fast algorithm for matching AIS trajectories with radar point data in complex environments'.

## installation
``` bash
pip install -r requirements.txt
```

## Usage
* To run the main.py script, you need to pass a series of parameters through the command line. Below is an example command that demonstrates how to set each parameter: *
``` bash 
python demo.py
```
* Parameter Descriptions: *
- `--batch_num`: Number of points in each batch of tracks.    
  The default value is 100.   
  
- `--radar_length`: Track length.    
  The default value is 20.  
  
- `--r`: Radius parameter for the Gaussian function.    
  The default value is 0.9.  
  
- `--r_threshold`: Filtering threshold.    
  The default value is 0.9.  
  
- `--threshold`: Threshold value.    
  The default value is 0.4.  
  
- `--resolution`: Resolution of the parameter space for voting.    
  The default value is 0.005.

* In demo.py, line 21, you can modify the generate_point function to generate different curves. *
