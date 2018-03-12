import tensorflow as tf
import random
import gym
import numpy as np
from gym.envs.reversi.reversi import ReversiEnv
import time
import os
from reversi_robot import reversi_robot
from model import CNN, Freezing_CNN, Simple_model

class RL_QG_agent:
    def __init__(self):
        # Init memory
        self.memory_size = 1024
        self.memory_cnt = 0
        self.memory = []

        self.loading_model = True
        #self.loading_model = False
        self.model_type = 'simple_model'
        #self.model_type = 'simple_cnn'
        #self.model_type = 'freezing_cnn'

        self.test_freq = 500
        self.test_game_cnt = 50
        self.best_score = 0

        self.play_game_times = 200000
        self.eps = 0.5
        self.eps_min = 0.01
        self.eps_decay = 0.999
        self.gamma = 0.99

        self.pre_train_step = 6000
        self.learn_freq = 3
        self.replace_model_freq = 5
        self.batch_size = 32
        self.learn_step = 0

        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi_model/best2018-01-24 20:06:21simple_model200000115")
        self.env = gym.make('Reversi8x8-v0')


        self.sess = tf.Session()
        self.init_model()

    def init_model(self):
        print(self.model_type)

        if os.path.exists(self.model_dir):
            pass
        else:
            os.makedirs(self.model_dir)

        if self.model_type == 'simple_model':
            self.model = Simple_model(env = self.env)
            if not self.loading_model:
                self.simple_train()
                #self.train()

        elif self.model_type == 'simple_cnn':
            self.model = CNN(env = self.env)
            if not self.loading_model:
                self.simple_train()
                # self.train()
        else:
            self.model = Freezing_CNN(env = self.env)
            if not self.loading_model:
                self.train(freezing = True)

        self.saver = tf.train.Saver()
        self.load_model()

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def remember(self, s, a, r, next_s, done, player):
        index = self.memory_cnt % self.memory_size
        if self.memory_cnt < self.memory_size:
            self.memory.append((s, a, r, next_s, done, player))
        else:
            self.memory[index] = (s, a, r, next_s, done, player)
        self.memory_cnt += 1

    def flat(self, s):
        return np.reshape(s, (1, self.model.input_length))

    def choose_action(self, state, enables, step):
        if np.random.rand(1) < self.eps or step < self.pre_train_step:
            return np.random.choice(enables)
        else:
            Q = self.sess.run(self.model.Q, \
            feed_dict={self.model.input_s: self.flat(state)})
            Q = np.ravel(Q)
            return enables[np.argmax(Q[enables])]

    def freezing_learn(self, game_i):
        self.learn_step += 1
        if self.learn_step % self.replace_model_freq == 0:
            self.sess.run(self.model.replace_model_op)
        batches = np.random.choice(len(self.memory), self.batch_size)
        for i in batches:
            s, a, r, next_s, done, player = self.memory[i]
            Q = self.sess.run(self.model.Q, feed_dict={self.model.input_s: self.flat(s)})

            if done:
                max_next_Q = 0
            else:
                next_Q = self.sess.run(self.model.freezing_Q, \
                                       feed_dict={self.model.input_s: self.flat(next_s)})
                next_Q_flatted = np.ravel(next_Q)
                next_enables = ReversiEnv.get_possible_actions(next_s, 1 - player)
                max_next_Q = np.max(next_Q_flatted[next_enables])

            if game_i == self.play_game_times - 1:
                print('Q', Q[0][a], end = "")
            Q_target = Q
            Q_target[0][a] = r + self.gamma * max_next_Q
            if game_i == self.play_game_times - 1:
                print(' Q_target{}'.format(Q_target[0][a]))

            # update
            loss, update = self.sess.run([self.model.loss, self.model.update], \
                                   feed_dict={self.model.Q_target: Q_target, self.model.input_s: self.flat(s)})
            #print('loss', loss)

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def learn(self):
        self.learn_step += 1
        batches = np.random.choice(len(self.memory), self.batch_size)
        for i in batches:
            s, a, r, next_s, done, player = self.memory[i]
            Q = self.sess.run(self.model.Q, feed_dict={self.model.input_s: self.flat(s)})

            if done:
                max_next_Q = 0
            else:
                next_Q = self.sess.run(self.model.Q, \
                                       feed_dict={self.model.input_s: self.flat(next_s)})
                next_Q_flatted = np.ravel(next_Q)
                next_enables = ReversiEnv.get_possible_actions(next_s, 1 - player)
                max_next_Q = np.max(next_Q_flatted[next_enables])

            Q_target = Q
            Q_target[0][a] = r + self.gamma * max_next_Q

            # update
            update = self.sess.run(self.model.update, \
             feed_dict={self.model.Q_target: Q_target, self.model.input_s: self.flat(s)})

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def __del__(self):
        print('sess close')
        self.sess.close()

    def place(self,state,enables,player):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        Q = self.sess.run(self.model.Q, \
                            feed_dict={self.model.input_s: np.reshape(state, (1, self.model.input_length))})
        Q = np.ravel(Q)
        action = enables[np.argmax(Q[enables])]
        return action

    def train(self, freezing = False):
        print('Start training with memory')
        self.sess.run(self.model.init)
        if freezing:
            # init latest_model
            self.sess.run(self.model.replace_model_op)

        step = 0
        for i in range(self.play_game_times):
            s = self.env.reset()
            player = 0
            step_in_a_game = 0
            while True:
                step_in_a_game += 1
                enables = self.env.possible_actions
                a = self.choose_action(s, enables, step)
                next_s, r, done, _ = self.env.step((a, player))

                self.remember(s, a, r, next_s, done, player)

                if (step > self.pre_train_step) and (step % self.learn_freq == 0):
                    if freezing:
                        self.freezing_learn(i)
                    else:
                        self.learn()

                if done:
                    #print('game {} : {}'.format(i, step_in_a_game))
                    break
                s = next_s
                player ^= 1
                step += 1

            # do test every test_freq games
            if i % self.test_freq == 0:
                score = 0
                for _ in range(3):
                    score += self.test(self.test_game_cnt)
                print("test ", i, " score: ", score / 3)
                if score > self.best_score:
                    self.best_score = score
                    saver = tf.train.Saver()
                    saver.save(self.sess, os.path.join(self.model_dir, 'best'
                                                       +time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                                                       +self.model_type + str(self.play_game_times)
                                                       +self.best_score
                                                       , 'parameter.ckpt'))

        # save model
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def simple_train(self):
        print('Start simple training')
        self.sess.run(self.model.init)
        for i in range(self.play_game_times):
            s = self.env.reset()
            step = 0
            player = 1
            robot_player = reversi_robot(self.env,
                                         self.sess,
                                         self.model,
                                         player)
            while True:
                step += 1
                # get Q
                Q = self.sess.run(self.model.Q,\
                                feed_dict={self.model.input_s: self.flat(s)})

                #enables = self.env.possible_actions
                enables = robot_player.get_possible_actions(s,player)

                a = robot_player.get_next_action(enables,Q)
                _s, r, done, _  = robot_player.step(a,player)

                #copy_env = copy.deepcopy(self.env)
                if not done:
                    # the opponent move(not change env)
                    opp_enables = robot_player.get_possible_actions(_s,1-player)
                    opp_a = self.place(_s, opp_enables, 1 - player)
                    next_s, r, done, _ = robot_player.step(opp_a, 1 - player)
                    if r == 1:
                        robot_player.update_Q(0,self.gamma)
                        print('black win game {} : {}'.format(i, step))
                        if self.eps > self.eps_min:
                            self.eps *= self.eps_decay
                        break
                    elif r == -1:
                        robot_player.update_Q(1,self.gamma)
                        print('whiter win game {} : {}'.format(i, step))
                        if self.eps > self.eps_min:
                            self.eps *= self.eps_decay
                        break
                else:
                    if r == 1:
                        robot_player.update_Q(0,self.gamma)
                        print('black win game {} : {}'.format(i, step))
                        if self.eps > self.eps_min:
                            self.eps *= self.eps_decay
                        break
                    elif r == -1:
                        robot_player.update_Q(1,self.gamma)
                        print('whiter win game {} : {}'.format(i, step))
                        if self.eps > self.eps_min:
                            self.eps *= self.eps_decay
                        break
                    else:
                        break

                #self.env = copy_env

                # if player == 1:
                #     r = -r
                #
                # enables = ReversiEnv.get_possible_actions(next_s, player)

                # get next_Q and find next_max_Q
                # if done:
                #     max_next_Q = 0
                # else:
                #     next_Q = self.sess.run(self.model.Q, \
                #                       feed_dict={self.model.input_s: self.flat(next_s)})
                #     next_Q_flatted = np.ravel(next_Q)
                #     max_next_Q = np.max(next_Q_flatted[enables])
                #
                # Q_target = Q
                # if i == self.play_game_times - 1:
                #     print('Q', Q[0][a])
                # Q_target[0][a] = r + self.gamma * max_next_Q
                # if i == self.play_game_times - 1:
                #     print('tQ ', Q_target[0][a])
                #
                # # update
                # _ = self.sess.run(self.model.update, \
                #         feed_dict={self.model.Q_target: Q_target, self.model.input_s: self.flat(s)})
                # s = _s
                #
                # # change player
                # player ^= 1
                # if done:
                #     print('game {} : {}'.format(i, step))
                #     if self.eps > self.eps_min:
                #         self.eps *= self.eps_decay
                #     break

            # do test every test_freq games
            if i and i % self.test_freq == 0:
                score = 0
                for _ in range(3):
                    score += self.test(self.test_game_cnt)
                print("test ", i, " score: ", score / (3*self.test_game_cnt))
                if score > self.best_score:
                    self.best_score = score
                    saver = tf.train.Saver()
                    saver.save(self.sess, os.path.join(self.model_dir, 'best'
                                                       +time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                                                       +self.model_type + str(self.play_game_times)
                                                       +str(self.best_score)
                                                       , 'parameter.ckpt'))

        # save model
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def test(self, play_game_times = 1000):
        env = gym.make('Reversi8x8-v0')
        env.reset()
        win_cnt = 0
        for i_episode in range(play_game_times):
            observation = env.reset()
            # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
            for t in range(100):
                action = [1,2]
                # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）
                ################### 黑棋 ############################### 0表示黑棋
                #  这部分 黑棋 是随机下棋
                enables = env.possible_actions
                if len(enables) == 0:
                    action_ = env.board_size**2 + 1
                else:
                    action_ = random.choice(enables)
                action[0] = action_
                action[1] = 0   # 黑棋 为 0
                observation, reward, done, info = env.step(action)
                ################### 白棋 ############################### 1表示白棋
                enables = env.possible_actions
                # if nothing to do ,select pass
                if len(enables) == 0:
                    action_ = env.board_size ** 2 + 1 # pass
                else:
                    action_ = random.choice(enables)
                    action_  = self.place(observation, enables,player = 1) # 调用自己训练的模型

                action[0] = action_
                action[1] = 1  # 白棋 为 1
                observation, reward, done, info = env.step(action)

                if done: # 游戏 结束
                    black_score = len(np.where(env.state[0,:,:]==1)[0])
                    white_score = len(np.where(env.state[1,:,:]==1)[0])
                    if black_score < white_score:
                        #print("白棋赢了！")
                        win_cnt += 1
                    break
        return win_cnt

if __name__ == '__main__':
    agent = RL_QG_agent()
