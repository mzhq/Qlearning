# -*- coding: utf-8 -*-
import numpy as np

class reversi_robot:

    status = {}
    enables = {}
    def __init__(self,env,sess,model,player,eps=0.5):
        self.env = env
        self.sess = sess
        self.player = player
        self.model = model
        self.eps = eps
        self.status[0] = []
        self.status[1] = []
        self.enables[0] = []
        self.enables[1] = []

    def fixed_reward(self,r):
        pass

        return r

    def step(self,enables,player):
        s,r,done,info = self.env.step((enables,player))
        if player == 1:
            self.status[1].append(s)
        else:
            self.status[0].append(s)
        r = self.fixed_reward(r)
        return s,r,done,info

    def get_next_action(self,enables,Q):
        if np.random.rand(1) < self.eps:
            a = np.random.choice(enables)
        else:
            Q_flatted = np.ravel(Q)
            a = enables[np.argmax(Q_flatted[enables])]
        return a

    def get_possible_actions(self,status,player):
        enable = self.env.get_possible_actions(status, player)
        if player == 1:
            self.enables[1].append(enable)
        else:
            self.enables[0].append(enable)
        return enable

    def flat(self, s):
        return np.reshape(s, (1, self.model.input_length))

    def update_Q(self,winner,gamma):
        Q = self.sess.run(self.model.Q,\
                    feed_dict={self.model.input_s: self.flat(self.status[winner][0])})
        # if winner == 1:  # 白棋赢
        #     r = 10
        # else:
        #     r = 10
        loser = 1- winner
        winner_r = 10
        loser_p = -10

        for i in range(len(self.status[winner])):
            if i == len(self.status[winner])-1: break
            next_Q = self.sess.run(self.model.Q, \
                                      feed_dict={self.model.input_s: self.flat(self.status[winner][i+1])})
            next_Q_flatted = np.ravel(next_Q)
            max_next_Q = np.max(next_Q_flatted[self.enables[winner][i]])
            Q_target = Q
            a = self.get_next_action(self.enables[winner][i],Q)
            Q_target[0][a] = winner_r + gamma * max_next_Q
            _ = self.sess.run(self.model.update, \
                    feed_dict={self.model.Q_target: Q_target,
                          self.model.input_s: self.flat(self.status[winner][i+1])})
            Q = self.sess.run(self.model.Q,\
                    feed_dict={self.model.input_s: self.flat(self.status[winner][i+1])})

        Q = self.sess.run(self.model.Q,\
                    feed_dict={self.model.input_s: self.flat(self.status[loser][0])})
        for i in range(len(self.status[loser])):
            if i == len(self.status[loser])-1: break
            next_Q = self.sess.run(self.model.Q, \
                                      feed_dict={self.model.input_s: self.flat(self.status[loser][i+1])})
            next_Q_flatted = np.ravel(next_Q)
            max_next_Q = np.max(next_Q_flatted[self.enables[loser][i]])
            Q_target = Q
            a = self.get_next_action(self.enables[loser][i],Q)
            Q_target[0][a] = loser_p + gamma * max_next_Q
            _ = self.sess.run(self.model.update, \
                    feed_dict={self.model.Q_target: Q_target,
                          self.model.input_s: self.flat(self.status[loser][i+1])})
            Q = self.sess.run(self.model.Q,\
                    feed_dict={self.model.input_s: self.flat(self.status[loser][i+1])})
