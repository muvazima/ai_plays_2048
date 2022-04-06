# ai-project

To play the 2048 game **manually**, run the following command from the project directory

    python Test/2048solver.py -s m

To use **Greedy Search**, run the following command:

    python Test/2048solver.py -s g

To use **Expectimax**, run the below command:

    python Test/2048solver.py -s e

Change the depth by updating line 14 of Expectimax.py


To use **Monte Carlo Tree Search**, run the below command:

    python Test/2048solver.py -s mc

Fix the directory and import error:
export PYTHONPATH="${PYTHONPATH}:/Users/smcck/Documents/CS-DS/Artificial Intelligence/ai-project"
export PYTHONPATH="${PYTHONPATH}:/Users/smcck/Documents/CS-DS/Artificial Intelligence/ai-project/game"
