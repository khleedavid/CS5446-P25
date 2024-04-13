# CS5446-Project

airPack_ws.ipynb: contains the base setup for getting the gym env running.

Todo:
- Find out how to modify the method for stacking
    - modify the constraints to add factors such as weight distribution
- Is it possible to load the items from the side instead of the top? How does that change the loading pattern?


REQUIREMENTS:
- Python 3.11
- gymnasium 
- gym_BinPack3D: you'll need to use a modified version of binpack to work with stable_baselines3 https://github.com/khleedavid/gym-BinPack3D
