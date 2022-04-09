from turtle import clear
from game_logic import *
from game_util_new import * 
from game_display import *
import numpy as np
import random


def basic_reward(game, action):
    next_game = game.copy()
    next_game.move(action)
    return next_game.score - game.score


def weightage(factor,i):
    return factor**i

# features:pairs

def feature_pairs(X,param):
    # 4 X 3 and 3 X 4 slicing 
    var3_x_4s = (weightage(param,1) * X[:3, :] +weightage(param,0) * X[1:, :]).ravel()
    var4_x_3s = (weightage(param,1)* X[:, :3] +weightage(param,0) * X[:, 1:]).ravel()
    return np.concatenate([var3_x_4s,var4_x_3s])


# features : triplets + corners

def feature_triplets(X,param):
    verticals = (weightage(param,2) * X[:2, :] +weightage(param,1) * X[1:3, :] +weightage(param,0)*X[2:, :]).ravel()
    horizontals = (weightage(param,2) * X[:, :2] +weightage(param,1) * X[:, 1:3] +weightage(param,0)*X[:, 2:]).ravel()
    X_except_top_left = (weightage(param,2) * X[1:, :3] +weightage(param,1) * X[1:, 1:] +weightage(param,0)*X[:3, 1:]).ravel()
    X_except_top_right = (weightage(param,2) * X[:3, :3] +weightage(param,1) * X[1:, :3] +weightage(param,0)*X[1:, 1:]).ravel()
    X_except_bottom_left = (weightage(param,2) * X[:3, :3] +weightage(param,1) * X[:3, 1:] +weightage(param,0)*X[1:, 1:]).ravel()
    X_except_bottom_right = (weightage(param,2) * X[:3, :3] +weightage(param,1) * X[1:, :3] +weightage(param,0)*X[:3, 1:]).ravel()
    return np.concatenate([verticals, horizontals, X_except_top_left, X_except_top_right, X_except_bottom_left, X_except_bottom_right])

#features: squares in quartets

def feature_quartets(X,param):
    verticals = (weightage(param,3) * X[0, :] +weightage(param,2)  * X[1, :] +weightage(param,1)  * X[2, :] +weightage(param,0) *X[3, :]).ravel()
    horizontals = (weightage(param,3) * X[:, 0] +weightage(param,2) * X[:, 1] +weightage(param,1) * X[:, 2] +weightage(param,0) * X[:, 3]).ravel()
    squares = (weightage(param,3) * X[:3, :3] +weightage(param,2) * X[1:, :3] +weightage(param,1) * X[:3, 1:] +weightage(param,0) * X[1:, 1:]).ravel()
    return np.concatenate([verticals,horizontals,squares])


class Q_agent:

    save_file = "agent.npy"     # saves the agent
    feature_functions = {2: feature_pairs, 3: feature_triplets, 4: feature_quartets}
    parameter_shape = {2: (24, 256), 3: (52, 4096), 4: (17, 65536)}
 
    # Discount, learning rate chosen as 1, 0.2 after multiple experiments.
    # Discount = 0.999 was v slow after 150 episodes, need to explore further.
    def __init__(self, weights=None, reward=basic_reward, step=0, alpha=0.2, discount=1, epsilon = 0.0001, file=None, n=4, mode = "a"):
        
        self.R = reward
        self.step = step
        self.alpha = alpha
        self.discount = discount
        self.file = file or Q_agent.save_file
        self.n = n
        self.num_feat, self.size_feat = Q_agent.parameter_shape[n] 
        self.mode = mode
        self.param = 16
        self.epsilon = epsilon

        if weights is None:
            self.weights = weights or np.random.random((self.num_feat, self.size_feat)) / 100
        else:
            self.weights = weights

    def save_agent(self, file=None):
        file = file or self.file
        arr = np.array([self.weights, self.step, self.alpha, self.n])
        np.save(file, arr)
        pass

    @staticmethod
    def load_agent(file=save_file, mode = 'a'):
        arr = np.load(file, allow_pickle=True)
        agent = Q_agent(weights=arr[0], mode = mode, step=arr[1], alpha=arr[2], n=arr[3])
        return agent

    def features(self, X,param):
        return Q_agent.feature_functions[self.n](X,param)

    def evaluate(self, state,mode):
        if mode == "b":
            features = self.features(state.row,self.param)
            return np.sum(self.weights[range(self.num_feat), features])
        else:
            game = state.copy()
            return get_total_heuristic(game.row)

    def update(self, state, dw):
        self.step += 1
        
        def _upd(X):
            features = self.features(X,self.param)
            for i, f in enumerate(features):
                self.weights[i, f] += dw

        X = state.row
        for _ in range(4):
            _upd(X)
            X = np.transpose(X)
            _upd(X)
            X = np.rot90(np.transpose(X))


    def flipCoin(self,p):
        r = random.random()
        return r < p
    
    def episode(self,mode):

        trial = Game()
        state, previous_value = None, 0
        while not trial.game_over():
            action, best_value = 0, -np.inf
            
            #Epsilon Tuning
            if self.flipCoin(self.epsilon):   # Random.
                action = actions.index(random.choice(actions))
                test = trial.copy()
                if test.move(action):
                    val = self.evaluate(test,mode = mode)
                    if val > best_value: 
                        best_value = val
            else:
                for direction in range(4): # Best value, action.
                    test = trial.copy()
                    if test.move(direction):
                        value = self.evaluate(test,mode = mode)
                        if value > best_value:
                            action, best_value = direction, value
            if state:
                #Bellman Equation
                game_reward = self.R(trial, action)
                dw = self.alpha * (game_reward + self.discount*best_value - previous_value) / self.num_feat
                self.update(state, dw)
            
            trial.move(action)
            state, previous_value = trial.copy(), best_value
            trial.new_tile()
        
        # Back propagation choices
        dw = - self.alpha * previous_value / self.num_feat #Back Propagation to update the feature weights
    
        self.update(state, dw)
        trial.history.append(trial)
        return trial

    @staticmethod
    def train_run(num_episodes, agent=None, file=None, start_ep=0, saving=True, mode = "a", display_option = 'y'):
        
        if agent is None:
            agent = Q_agent(mode = mode)
        if file:
            agent.file = file
        
        average_over_1000_episodes = []
        moving_average = []

        reached = [0] * 7
        best_game, best_score = None, 0
        start = time.time()
        
        for i in range(start_ep + 1, num_episodes + 1):
            game = agent.episode(mode)
            
            if(game.score % 2 != 0 ): 
                print("WTF")
            
            print("GAME SCORE:", game.score)
            
            moving_average.append(game.score)
            average_over_1000_episodes.append(game.score)
            
            if game.score > best_score:
                best_game, best_score = game, game.score
                print('new best game!')
                print(game,type(game))

                # Shows current best scores during training.
                if display_option == 'y':
                    display = Display(matrix = game.row)

                
                if saving:
                    game.save_game(file = 'best_game.npy')
                    print('game saved at best_game.npy')
                    agent.save_agent(file = 'best_agent.npy')
                    print(f'agent saved in {agent.file}')
            
            max_tile = np.max(game.row)
            
            if max_tile >= 10:
                reached[max_tile - 10] += 1
            
            # Moving average after every episode.
            print(i, game.odometer, game.score, 'reached', 1 << np.max(game.row), '100-ma=', int(np.mean(moving_average)))
            
            # Saving after every 100 steps. 
            if saving and i % 100 == 0:
                agent.save_agent()
                print(f'agent saved in {agent.file}')
            
            # Average over last 1000 episodes.
            if i % 1000 == 0:
                print('------')
                print((time.time() - start) / 60, "min")
                start = time.time()
                print(f'episode = {i}')
                print(f'average over last 1000 episodes = {np.mean(average_over_1000_episodes)}')
                average_over_1000_episodes = []
                
                for j in range(7):
                    r = sum(reached[j:]) / 10
                    print(f'{1 << (j + 10)} reached in {r} %')
                
                reached = [0] * 7
                print(f'best score so far = {best_score}')
                print(best_game)
                print(f'current learning rate = {agent.alpha}')
                print('------')


if __name__ == "__main__":

    agent = Q_agent(n=4, reward=basic_reward, alpha=0.1, file = "new_agent.npy")
    # agent = Q_agent.load_agent(file="best_agent.npy")

    Q_agent.train_run(num_episodes = 100000, agent=agent, file = "new_best_agent.npy", start_ep=0)
