# DRL-Agent

Naive DQN & Double DQN & Dueling DQN & A2C <br>
This project mainly used to compare the performance of DQN-based algorithm and A2C algorithm in several games, include ['CartPole-v0', 'LunarLander-v2', 'Breakout-v0', 'CarRacing-v0', 'Pendulum-v0'] 

I will add the result score curve diagram and how to run it in the later time (After I get a job 🤣 )

## how to run

Requires Pytorch 1.7+ and Python 3.7 to run.<br>
Install the requirements from requirements.txt before start.

train the game by DQN-based alg, -gidx donates game index, -dqn donates DQN type
```sh
cd DQN
python main.py -mode train -gidx 0 -dqn 0 -step 5000 -lr 1e-3
```
run the model
```sh
cd DQN
python main.py -mode run -gidx 0 -dqn 0
```
