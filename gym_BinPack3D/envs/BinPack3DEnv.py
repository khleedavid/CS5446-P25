# import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
# replace gym with gymnasium
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

import numpy as np
import copy

from gym_BinPack3D.envs.Container import Container, Box
from gym_BinPack3D.envs.BoxSeqGenerator import BoxSeqGenerator, RandomBoxCreator, CuttingBoxCreator, Rotate


class PackingGame(gym.Env):
    """
    x: depth ( small x = deep inside, large x = near to viewer)
    y: length (small y = left       , large y = right)
    z: height (small z = low        , large z = high)
       Z
       |
       |
       |______Y
      /
     /
    X

    Concept:
    box: you pack boxes into a container
    container: contain boxes
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(   self, 
                    container_size = (20, 20, 20),
                    boxSeqGenerator='random', 
                    enabled_rotations = [Rotate.NOOP],
                    n_foreseeable_box = 1,
                    box_set = [Box(1,1,1), Box(2,3,4)], 
                    minSideLen = None,
                    maxSideLen = None,
                    genValidPlacementMask = True,
                    render_mode = "rgb_array",
                    #data_name = None,  #TODO: load saved box seq
                    # **kwargs
                    ):
        """
        Caveat: order in list "enabled_rotations" affects action meaning.
        below should work, other orders probably not
        [Rotate.NOOP]
        [Rotate.NOOP, Rotate.XY]
        [Rotate.NOOP, Rotate.XY, Rotate.XZ]
        [Rotate.NOOP, Rotate.XY, Rotate.XZ, Rotate.YZ]
        """
        self.render_mode = render_mode

        self.container_size = container_size
        self.container_area = int(self.container_size[0] * self.container_size[1])
        self.container_vol  = int(self.container_size[0] * self.container_size[1] * self.container_size[2])
        self.container = Container(*self.container_size)

        self.box_set = box_set
        self.enabled_rotations = enabled_rotations
        self.n_foreseeable_box = n_foreseeable_box

        self.boxSeqGenerator = boxSeqGenerator
        if type(boxSeqGenerator) is str:
            assert box_set is not None
            if boxSeqGenerator == 'random':
                print('using random box sequence')
                self.boxSeqGenerator = RandomBoxCreator(box_set, self.enabled_rotations, n_foreseeable_box)

            elif boxSeqGenerator == 'CUT-1':
                print('using CUT-1 logic box sequence')
                self.boxSeqGenerator = CuttingBoxCreator(container_size, minSideLen, maxSideLen, "ByZ",
                                                         self.enabled_rotations, n_foreseeable_box
                                                         )
            elif boxSeqGenerator == 'CUT-2':
                print('using CUT-2 logic box sequence')
                self.boxSeqGenerator = CuttingBoxCreator(container_size, minSideLen, maxSideLen, "ByStackOrder",
                                                         self.enabled_rotations, n_foreseeable_box,
                                                         )
        assert isinstance(self.boxSeqGenerator, BoxSeqGenerator)    

        self.genValidPlacementMask = genValidPlacementMask

        obsSpace = {
            "height_map"   : spaces.Box(low=0.0, high=self.container_size[2], shape=(self.container_size[0],self.container_size[1]) ),
            "coming_boxes" : spaces.Box(low=0.0, high=max(self.container_size), shape=(self.n_foreseeable_box,3) ),
        }

        if self.genValidPlacementMask:
            obsSpace["valid_placement_mask"] = spaces.MultiBinary( [len(self.enabled_rotations), self.container_size[0],self.container_size[1] ])

        self.action_space = spaces.MultiDiscrete( [self.container_area, len(self.enabled_rotations)] )
        self.observation_space = spaces.Dict(obsSpace)
        

    def _get_obs(self):
        return self.cur_observation
    
    def _get_info(self):
        return {'counter':len(self.container.boxes), 'ratio':self.container.get_fill_ratio()}

    #def get_box_ratio(self):
    #    coming_box = self.next_box
    #    return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])


    #def get_box_plain(self):
    #    x_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[0]
    #    y_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[1]
    #    z_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[2]
    #    return (x_plain, y_plain, z_plain)
    
    def actionIdx_to_position(self, idx):
        lx = idx // self.container_size[1]
        ly = idx % self.container_size[1]
        return (lx, ly)

    def position_to_actionIdx(self, pos):
        assert len(pos) == 2
        assert pos[0] >= 0 and pos[1] >= 0
        assert pos[0] < self.container_size[0] and pos[1] < self.container_size[1]
        return pos[0] * self.container_size[1] + pos[1]

    @property
    def cur_observation(self):
        hmap = self.container.heightMap
        coming_boxes = self.boxSeqGenerator.next_N_boxes()
        firstBox = coming_boxes[0]
        coming_boxes = np.array([(b.dx,b.dy,b.dz) for b in coming_boxes], dtype=int)

        if self.genValidPlacementMask:
            mask = []
            for r in self.enabled_rotations:
                b = copy.deepcopy(firstBox)
                b.rotate(r)
                mask.append( (self.container.get_possible_positions(b)>0).astype(np.int8) )
            mask = np.array(mask)

        obs =  {
                "height_map"   : hmap,
                "coming_boxes" : coming_boxes
               }
        if self.genValidPlacementMask: obs["valid_placement_mask"] = mask
        
        return obs

    def step(self, action):
        position = self.actionIdx_to_position(action[0])
        rotation = self.enabled_rotations[action[1]]
        box = self.boxSeqGenerator.next_N_boxes()[0]
        box.rotate(rotation)

        succeeded = self.container.drop_box(box, position)

        if succeeded:
            self.boxSeqGenerator.pop_box()  # Remove current box from the sequence
            
            base_reward = ((box.dx * box.dy * box.dz) / self.container_vol) * 10  # Adjust the scale factor as needed

            # Encourage starting from the back of the container and filling towards the front
            x_penalty = (box.x / self.container.dx) * 10  # Increase penalty as x increases

            space_usage = self.container.get_usage_ratio()
            space_usage_reward = (space_usage) * 10  # Encourage using more space

            #reward longer sequences of boxes placed in the container. we want to maximize the number of boxes placed in the container.
            curr_count = len(self.container.boxes)
            # larger count is better. use sigmoid to reward the agent for placing more boxes. limit the range of the reward to -1 to 1.
            reward_count = ((1+np.exp(-curr_count))**-1) * 10

            reward = space_usage_reward + reward_count - x_penalty + base_reward

            done = False
        else:
            reward = -100  # Large penalty for failing to place a box
            done = True

        info = {
            'counter': len(self.container.boxes),
            'ratio': self.container.get_fill_ratio(),
            'space_usage': self.container.get_usage_ratio(),
            'last_position': position,
            'last_rotation': rotation
        }

        truncated = False
        return self.cur_observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.boxSeqGenerator.reset(seed=seed, options=options)
        self.container.reset(seed=seed)

        return self.cur_observation, self._get_info()

    def render(self, mode= None):
        from gym_BinPack3D.envs.VisUtil import plot_box
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        iMode = plt.isinteractive()

        if mode is None:
            mode = self.render_mode

        if mode=='human':
            plt.interactive(True)
        if mode=='rgb_array':
            plt.interactive(False)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        maxSideLen = max(self.container_size)
        box = Box(maxSideLen,maxSideLen,maxSideLen)
        plot_box(box, ax, color=(0,0,0,0), showEdges=False) # invisible bound box
        
        box = Box(*self.container_size)
        plot_box(box, ax)
        
        for box in self.container.boxes[:-1]:
            plot_box(box, ax, color=(0,0.5,1,1))
        
        if len(self.container.boxes)>0:
            box = self.container.boxes[-1]
            plot_box(box, ax, color=(0,0.,1,1)) # plot the latest box in diff color

        iMode = plt.interactive(iMode)  #restore
        if mode=='human':
            return fig
        if mode=='rgb_array':
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image_from_plot

        return None

    def close(self):
        pass

