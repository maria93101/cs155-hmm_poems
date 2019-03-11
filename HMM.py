########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]


        # Initialize transition probabilities based on the starting
        # transition probablities A_start and the observation matrix O
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]

        for i in range(1, M): # For each observation
            for j in range(self.L): # For each state
                max_like = 0
                max_state = 0
                for l in range(self.L):
                # calculate the sum of the probabilities of having been in every state
                # before, multiplied by the state-transition probability
                    temp = probs[i][l] * self.A[l][j] * self.O[j][x[i]]
                    if (temp >= max_like):
                        max_like = temp
                        max_state = l
                probs[i + 1][j] = max_like
                seqs[i + 1][j] = max_state

        max_el = max(probs[M])
        index = probs[M].index(max_el)

        max_seq = str(index)
        # build the output sequence with all the max pointers
        for i in range(M + 1, 1, -1):
            index = int(max_seq[len(max_seq) - 1])
            max_seq += str(seqs[i - 1][index])

        return max_seq[::-1]


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize
        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for i in range(1, M):
            for j in range(self.L):
                sum = 0
                for l in range(self.L):
                    sum += alphas[i][l] * self.A[l][j]
                alphas[i + 1][j] = self.O[j][x[i]] * sum

        if (normalize):
            for i in range(1, M):
                sum = 0
                for a in alphas[i + 1]:
                    sum += a
                if sum != 0:
                    alphas[i + 1] = [a / sum for a in alphas[i + 1]]

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            betas[M][i] = 1

        for i in range(M - 1, -1, -1):
            for j in range(self.L):
                sum = 0
                for l in range(self.L):
                    sum += betas[i + 1][l] * self.A[j][l] * self.O[l][x[i]]
                betas[i][j] = sum

        if (normalize):
            for i in range(M + 1):
                norm = 0
                for b in betas[i]:
                    norm += b
                if norm != 0:
                    betas[i] = [b / norm for b in betas[i]]

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        # Initialize A
        self.A = [[0 for i in range(self.L)] for j in range(self.L)]
        A_divide = [0 for i in range(self.L)] # for counting

        for i in range(len(Y)): # for each sequence
            for j in range(len(Y[i]) - 1): # for each thing in the sequence
                seq = Y[i]
                self.A[seq[i]][seq[i + 1]] += 1
                A_divide[seq[i]] += 1

        # now divide by count of occurances
        for j in range(self.L):
            for i in range(self.L):
                self.A[j][i] = self.A[j][i] / A_divide[j]

        # Calculate each element of O using the M-step formulas.

        self.O = [[0 for i in range(self.D)] for j in range(self.L)]
        O_divide = [0 for i in range(self.L)]

        for i in range(len(Y)): # for each sequence
            for j in range(len(Y[i]) - 1): # for each thing in the sequence
                self.O[Y[i][j]][X[i][j]] += 1
                O_divide[seq[i]] += 1

        # now divide by count of occurances
        for j in range(self.L):
            for i in range(self.D):
                self.O[j][i] = self.O[j][i] / O_divide[j]

        return


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        # randomly initialize HMM model parameters

        for n in range(N_iters):
            print("merp " + str(n))
            P_ya_t = []
            P_yab_t = []

            for xi, x in enumerate(X):
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                M = len(x)

                P_ya = [[0 for i in range(self.L)] for j in range(M)]
                P_yab = [[[0 for i in range(self.L)] for i in range(self.L)] for j in range(M)]

                # calculate first marginal
                for t in range(M):

                    norm = 0
                    for k in range(self.L):
                        temp = alphas[t + 1][k] * betas[t + 1][k]
                        P_ya[t][k] = temp
                        norm += temp

                    # now normalize
                    if norm != 0:
                        for l in range(self.L):
                            P_ya[t][l] = P_ya[t][l] / norm

                # calculate second marginal
                for t in range(M - 1):

                    norm = 0
                    for j in range(self.L):
                        for k in range(self.L):
                            temp =  alphas[t + 1][j] * self.A[j][k] * \
                                    self.O[k][x[t + 1]] * betas[t + 2][k]
                            P_yab[t][j][k] = temp
                            norm += temp

                    # now normalize
                    if norm != 0:
                        for j in range(self.L):
                            for k in range(self.L):
                                P_yab[t][j][k] = P_yab[t][j][k] / norm

                P_yab_t.append(P_yab)
                P_ya_t.append(P_ya)

            # Now update the values of A and O
            for i in range(self.L):
                for j in range(self.L):
                    top = 0
                    bottom = 0
                    for x in range(len(X)):
                        M = len(X[x])
                        for t in range(M - 1):
                            top += P_yab_t[x][t][i][j]
                            bottom += P_ya_t[x][t][i]
                    if bottom != 0:
                        self.A[i][j] = top / bottom

            for i in range(self.L):
                for j in range(self.D):
                    top = 0
                    bottom = 0
                    for x in range(len(X)):
                        M = len(X[x])
                        for t in range(M):
                            if X[x][t] == j: # indicator function
                                top += P_ya_t[x][t][i]
                            bottom += P_ya_t[x][t][i]
                    if bottom != 0:
                        self.O[i][j] = top / bottom

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        # add the starting state
        states.append(random.choice(range(self.L)))

        for i in range(M):

            rand = random.uniform(0, 1)
            state = 0
            while rand > 0:
#                 print (i)
                rand = rand - self.A[states[i]][state]
                state += 1
            states.append(state - 1)

            rand = random.uniform(0, 1)
            observation = 0
            while rand > 0:
                rand = rand - self.O[states[i]][observation]
                observation += 1
            emission.append(observation - 1)

# #             print(states)
# #             print (states[1])
#             #break

        return emission, states

    def generate_emission_syllables(self, M, obs_map_r, syllable_dict):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.   sjfnvln

        Arguments:
            M:          Length of the emission (# syllables) to generate (ex. 10).

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []


        # add the starting state
        states.append(random.choice(range(self.L)))

        i = 0
        while i < M:
            # print (i)

            rand = random.uniform(0, 1)
            state = 0
            while rand > 0:
                #  print (state, i)
                rand = rand - self.A[states[len(emission) - 1]][state]
                state += 1
            #states.append(state - 1)

            rand = random.uniform(0, 1)
            observation = 0
            while rand > 0:
                rand = rand - self.O[states[len(emission) - 1]][observation]
                observation += 1
            #emission.append(observation - 1)

            f = 0
            try:
                if syllable_dict[obs_map_r[observation-1]]:
                    num_syll = syllable_dict[obs_map_r[observation-1]]
                    if len(num_syll) == 1:
                        if int(num_syll[0]) == 1:
                            i += 1
                            f = 1
                        else:
                            if (i+int(num_syll[0]) > M):
                                # select something else
                                pass
                            else:
                                i += int(num_syll[0])
                                f = 1
                    else: # len(num_syll) > 1:
                        if 'E' in sorted(num_syll)[1]:
                            n = int(sorted(num_syll)[1][1:])
                            if i+n == M:
                                i += n
                                f = 1
                            elif i+int(sorted(num_syll)[0]) < M:
                                i+= int(sorted(num_syll)[0])
                                f = 1
                        else:
                            if i+int(num_syll[0]) <= M:
                                i+= int(num_syll[0])
                                f = 1
                            elif i+int(num_syll[1]) <= M:
                                i+= int(num_syll[1])
                                f = 1
                            else:
                                # select something else
                                pass
            except KeyError:
                # sadl
                # print (obs_map_r[observation-1])
                pass
            if (f):
                emission.append(observation - 1)
                states.append(state - 1)
# #             print(states)
# #             print (states[1])
#             #break

        return emission, states

    def generate_emission_syllables_other(self, M, obs_map_r, syllable_dict):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.   sjfnvln

        Arguments:
            M:          Length of the emission (# syllables) to generate (ex. 10).

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []


        # emission = x value (0 to D-1)
        # state = y value (0 to L-1)
        emission_vals = [d for d in range(self.D)]
        state_vals = [l for l in range(self.L)]

        # pick a random state
        state_start = random.randint(0, (self.L-1))

        # do this M times
        i = 0
        while i < M:
            # find state with weighted probabilities
            # use transmission matrix A and state_start
            if i == 0:
                A_probs = self.A[state_start]
            else:
                A_probs = self.A[states[len(emission) - 1]]
            state = random.choices(state_vals, A_probs)
            states.append(state[0])

            # find emission with weighted probabilities
            # use observation matrix O and current ? state
            O_probs = self.O[states[len(emission)]]
            em = random.choices(emission_vals, O_probs)

            f = 0
            try:
                if syllable_dict[obs_map_r[em[0]]]:
                    num_syll = syllable_dict[obs_map_r[em[0]]]
                    if len(num_syll) == 1:
                        if int(num_syll[0]) == 1:
                            i += 1
                            f = 1
                        else:
                            if (i+int(num_syll[0]) > M):
                                # select something else
                                pass
                            else:
                                i += int(num_syll[0])
                                f = 1
                    else: # len(num_syll) > 1:
                        if 'E' in sorted(num_syll)[1]:
                            n = int(sorted(num_syll)[1][1:])
                            if i+n == M:
                                i += n
                                f = 1
                            elif i+int(sorted(num_syll)[0]) < M:
                                i+= int(sorted(num_syll)[0])
                                f = 1
                        else:
                            if i+int(num_syll[0]) <= M:
                                i+= int(num_syll[0])
                                f = 1
                            elif i+int(num_syll[1]) <= M:
                                i+= int(num_syll[1])
                                f = 1
                            else:
                                # select something else
                                pass
            except KeyError:
                # sadl
                # print (obs_map_r[observation-1])
                pass
            if (f):
                emission.append(em[0])
                #states.append(state - 1)
            else:
                del state[-1]

        return emission, states

    def generate_emission_syllables_correct(self, M, obs_map_r, syllable_dict):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.   sjfnvln

        Arguments:
            M:          Length of the emission (# syllables) to generate (ex. 10).

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        state = random.choice(range(self.L))
        states = []

        i = 0
        while i < M:
            # Append state.
            states.append(state)

            # Sample next observation.
            rand_var = random.uniform(0, 1)
            next_obs = 0

            while rand_var > 0:
                rand_var -= self.O[states[len(emission)]][next_obs]
                next_obs += 1

            next_obs -= 1

            f = 0
            try:
                if syllable_dict[obs_map_r[next_obs]]:
                    num_syll = syllable_dict[obs_map_r[next_obs]]
                    if len(num_syll) == 1:
                        if int(num_syll[0]) == 1:
                            i += 1
                            f = 1
                        else:
                            if (i+int(num_syll[0]) > M):
                                # select something else
                                pass
                            else:
                                i += int(num_syll[0])
                                f = 1
                    else: # len(num_syll) > 1:
                        if 'E' in sorted(num_syll)[1]:
                            n = int(sorted(num_syll)[1][1:])
                            if i+n == M:
                                i += n
                                f = 1
                            elif i+int(sorted(num_syll)[0]) < M:
                                i+= int(sorted(num_syll)[0])
                                f = 1
                        else:
                            if i+int(num_syll[0]) <= M:
                                i+= int(num_syll[0])
                                f = 1
                            elif i+int(num_syll[1]) <= M:
                                i+= int(num_syll[1])
                                f = 1
                            else:
                                # select something else
                                pass
            except KeyError:
                # sadl
                # print (obs_map_r[observation-1])
                pass
            if (f):
                emission.append(next_obs)
                #states.append(state - 1)
            else:
                del states[-1]

            # Sample next state.
            rand_var = random.uniform(0, 1)
            next_state = 0

            while rand_var > 0:
                rand_var -= self.A[states[len(emission)-1]][next_state]
                next_state += 1

            next_state -= 1
            state = next_state

        return emission, states

    def generate_emission_syllables_rhymes(self, M, obs_map_r, syllable_dict, rhyme_dict):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission (# syllables) to generate (ex. 10).
            obs_map_r   Dictionary mapping encoded words (0:D-1) to English
            syllable_dict Dictionary mapping English words to number of syllables
            rhyme_dict Dictionary mapping English words to rhyming words

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        pass

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    random.seed(2019)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
