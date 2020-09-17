# Free-floating Electrical Vehicles Sharing Systems (FFEVSS)
This repository contains all implementations of "A Reinforcement Learning Approach for Rebalancing Electric Vehicle
Sharing Systems", inclduing environments for both a single and multi shuttle routing in FFEVSS. 

## Dependencies 
Pytorch 1. 5 
Python > 3.6 

## Content 
1. **agents** stores A2CAgents designed to train and test reinforcement learning agents for a single and multi shuttle routing, that use A2C algorithm for training . 
2. **data** containes test dataset designed for a specific urban network structure. 
3. envs stores two simulatores to represent environments of FFEVSS with a single and multiple shuttles and data generators that produce training data for simulators on the fly. 
4. **neuralnets** includes Actor and Critic neural nets that are based on sequence to sequence models with attention mechanism 
5. **results** stores inference results based on test datasest stored in the data folder 
6. **trained_models** containes keys of trained models and can be directly loaded to Actor and Critic models 
7. **main.py** imports all needed libraries either for a single or multi shuttle routing and allows to perform training and testing using specific RL agents.
8. **network_settings.py** specifies an urban network structure of FFEVSS that are later passed to data generator and environment. 
9. **options.py** inlcudes the argument values used throughout training and testing. 

## To Run 
1. Clone this repository. 
2. Specify an urban network structure of FFEVSS or use the default structure included in network_settings.py. 
3. Run python3 main.py, the default setting will start training for a single shuttle routing. 
4. To perform only inference please set 'train' to False in options.py 

## To Cite 
Please include citing info

## Sample Solutions 
A sample solution for a sinlge shuttle routing in the urban network of size 23 nodes. 
[example-single.pdf](https://github.com/aigerimb/FFEVSS/files/5240877/example-single.pdf)
A sample solution for a multi shuttle routing in the urban network of size 23 nodes. 
[samplefigure.pdf](https://github.com/aigerimb/FFEVSS/files/5240878/samplefigure.pdf)
