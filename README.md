# PM25Sim

##Data Generation
You can find the method for the data generation in [Data\Generators\AdDifGenerator.py](https://github.com/fabiuskt/PM25Sim/blob/master/Data/Generators/AdDifGenerator.py).
The method for the contamination is called true solution.
The method for the sourceterm is called source.
For generating Quadruples of Data (contamination, x-Coordinate, y-Coordinate, time) you can use Generate Equal. The Datapoints are on a regular grid.

##Prediction with NN
The neural network I am currently using is in [Src\pollutionModel](https://github.com/fabiuskt/PM25Sim/blob/master/Src/PollutionModel.py)
At the beginning you can specifiy how many data points and in which domain you want to generate.

Unfortunatly I had no time to make further tests or comment everything out. Sorry for that :(

