# _*_ coding:utf-8 _*_
import gym
import simple_dqn

ENV_NAME = "CartPole-v0"
EPISODE = 10000
STEP = 300
TEST = 10


def main():
    env = gym.make(ENV_NAME)
    agent = simple_dqn.SIMPLE_DQN(env)

    for episode in range(EPISODE):
        state = env.reset()

        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
            if ave_reward >= 200:
                break


if __name__ == '__main__':
    main()
