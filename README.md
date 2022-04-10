# ai-project
In this project, different AI algorithms are trained to play
2048, the single player tile-based puzzle game, and their performances
are compared. The algorithms implemented are - Greedy Search as the
baseline, Expectimax, Monte Carlo Tree Search and Deep Q-Learning.
The Performance of these algorithms were compared based on their exe-
cution time, highest score achieved and percentage of wins. Expectimax
was found to be the best performing, achieving a high score of 8192.

## How to Run
To play the 2048 game **manually**, run the below command from the project directory

    python Test/2048solver.py -s m

### Greedy Search
To use **Greedy Search**, run the below command:

    python Test/2048solver.py -s g

### Expectimax
To use **Expectimax**, run the below command:

    python Test/2048solver.py -s e

Change the depth by updating line 14 of Expectimax.py

### Monte Carlo Tree Search
To use **Monte Carlo Tree Search**, run the below command:

    python Test/2048solver.py -s mc

### Tuple based Q learning
To train the **RL agent**, run the below command:

    python3 rl_agent/training.py --mode b --display n --num 500 

Where --mode b = Tuple Heuristic, --mode a = Snake Heuristic , --display n/y , y: display grid whenever a best score is recorded, n: don't display grid and num = number of traiing episodes.

To use the trained **Q learning**, run the below command:
    python3 test.py --mode b --trials 15 

Where --trials = number of test episodes/game-plays

## References
[1]https://github.com/kiteco/python-youtube-code/tree/master/AI-plays-2048
[2]https://github.com/huntermills707/2048MDPsolver
[3]https://github.com/abachurin/2048


Fix the directory and import error:

`export PYTHONPATH="${PYTHONPATH}:/Users/smcck/Documents/CS-DS/Artificial Intelligence/ai-project"`

`export PYTHONPATH="${PYTHONPATH}:/Users/smcck/Documents/CS-DS/Artificial Intelligence/ai-project/game"`