# Free-floating Electrical Vehicles Sharing Systems (FFEVSS)
This repository contains all implementations of ["A Reinforcement Learning Approach for Rebalancing Electric Vehicle
Sharing Systems"](https://arxiv.org/abs/2010.02369), inclduing environments for both a single and multi shuttle routing in FFEVSS. 

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
3. Run main.py as follows with the default settings to train a single shuttle routing:
```bash
python main.py 
```
4. To perform only inference please set 'train' to False in options.py or just run:
```bash
python main.py --train=False
```

## To Cite 
```
@misc{bogyrbayeva2020reinforcement,
      title={A Reinforcement Learning Approach for Rebalancing Electric Vehicle Sharing Systems}, 
      author={Aigerim Bogyrbayeva and Sungwook Jang and Ankit Shah and Young Jae Jang and Changhyun Kwon},
      year={2020},
      eprint={2010.02369},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Sample Solutions 
A sample solution for a sinlge shuttle routing in the urban network of size 23 nodes. 
![example-single-1](https://user-images.githubusercontent.com/25514362/93505944-8c1cfd80-f8e9-11ea-81af-ae5d10f5eeaf.png)
A sample solution for a multi shuttle routing in the urban network of size 23 nodes. 
![samplefigure-1](https://user-images.githubusercontent.com/25514362/93505946-8cb59400-f8e9-11ea-901f-f5c9ca80e185.png)


