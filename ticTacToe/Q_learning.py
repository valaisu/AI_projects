import pygame
import numpy as np

class TicTacToeGUI:
    def __init__(self, size=3, win_length=3, training = False):
        self.size = size
        self.win_length = win_length
        self.board = [['' for _ in range(size)] for _ in range(size)]
        self.current_turn = 'X'
        self.moves_played = 0
        self.game_over = False
        self.cell_size = 100
        self.training = training

        # Initialize Pygame
        pygame.init()
        self.width = self.cell_size * size
        self.height = self.cell_size * size
        self.line_color = (0, 0, 0)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")
        self.screen.fill((255, 255, 255))
        if not self.training:
            self.draw_board()

        # Q learning
        self.observation_space_size = 3 ** (self.size ** 2)
        self.action_space_size = self.size ** 2
        self.q_table_1 = np.zeros((self.observation_space_size, self.action_space_size)) # start AI
        self.q_table_2 = np.zeros((self.observation_space_size, self.action_space_size)) # second AI
        self.epsilon = 1.0 # full exploration

    def draw_board(self):
        """
        Draws an empty board
        :return: None
        """
        for x in range(1, self.size):
            pygame.draw.line(self.screen, self.line_color, (x * self.cell_size, 0), (x * self.cell_size, self.height), 2)
            pygame.draw.line(self.screen, self.line_color, (0, x * self.cell_size), (self.width, x * self.cell_size), 2)

    def mark_square(self, row, col, draw = False):
        """
        Marks move on internal memory
        :param row: int
        :param col: int
        :param draw: Bool
        :return: None
        """
        if self.board[row][col] == '' and not self.game_over:
            self.board[row][col] = self.current_turn
            if draw:
                self.draw_mark(row, col)
            if self.check_winner(row, col):
                self.game_over = True
            else:
                self.current_turn = 'O' if self.current_turn == 'X' else 'X'
                self.moves_played += 1

    def draw_mark(self, row, col):
        """
        Draws a move on the board
        :param row:
        :param col:
        :return: None
        """
        x = col * self.cell_size
        y = row * self.cell_size
        if self.current_turn == 'X':
            pygame.draw.line(self.screen, (255, 0, 0), (x + 15, y + 15), (x + self.cell_size - 15, y + self.cell_size - 15), 15)
            pygame.draw.line(self.screen, (255, 0, 0), (x + self.cell_size - 15, y + 15), (x + 15, y + self.cell_size - 15), 15)
        else:
            pygame.draw.circle(self.screen, (0, 255, 0), (x + self.cell_size // 2, y + self.cell_size // 2), self.cell_size // 2 - 15, 15)

    def check_winner(self, row, col):
        """
        Inputs the last move, checks squares to every direction.
        If enough squares in row are the same, the move was winning
        :param row: int
        :param col: int
        :return: None
        """
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for direction in directions:
            count = 0
            for i in range(1, self.win_length):
                if row + i * direction[0] < 0 or row + i * direction[0] >= self.size or col + i * direction[1] < 0 or col + i * direction[1] >= self.size:
                    break
                if self.board[row + i * direction[0]][col + i * direction[1]] == self.current_turn:
                    count += 1
                else:
                    break
            if count == self.win_length - 1:
                return True

    def run_game(self):
        """
        Lets the user play for both teams
        :return:
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    mouseX = event.pos[0] // self.cell_size
                    mouseY = event.pos[1] // self.cell_size
                    self.mark_square(mouseY, mouseX)
                    [print(i) for i in self.one_hot_encode()]

            pygame.display.update()
        pygame.quit()

    def sample_actions(self, actions, turn):
        """
        Takes actions,
        if self.epsilon is high is likely to explore/take random actions
        if self.epsilon is low is likely to choose the best action
        :param actions: list[int]
            indices of the possible actions
        :param turn: int
            tells which q-table to use
            1 -> start
            0 -> second
        :return: int
            The action taken
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        else:
            if turn == 1:
                col = self.q_table_1[self.state_to_index()]
                valid = [(col[i], i) for i in actions]
                best = max(valid)[1]
                return best#np.argmax(self.q_table_1[self.state_to_index()])
            else:
                col = self.q_table_2[self.state_to_index()]
                valid = [(col[i], i) for i in actions]
                best = max(valid)[1]
                return best

    def train(self, iterations, learning_rate = 0.7, discount_rate = 0.5, decay_rate_normalized = 1):
        # Best params so far: .7, .5, 1
        decay_rate = decay_rate_normalized / (iterations * 5)
        self.epsilon = 1.0
        for episode in range(iterations):
            prev_states = [self.state_to_index(), self.state_to_index()]
            #                   second                   first
            t = 1
            prev_move = (0, 0)
            while not self.game_over:

                update = self.q_table_1 if self.current_turn == 'X' else self.q_table_2
                action_space = self.action_space_get()
                if len(action_space) == 0:
                    break
                action = self.sample_actions(action_space, t)
                new_state, reward, game_over = self.step(action)
                update[prev_states[t]][action] = update[prev_states[t]][action] + learning_rate * (reward[t] + discount_rate * np.max(update[new_state]) - update[prev_states[t]][action])
                # reward of prev action: Q[prev][action] = Q[prev][action] + alpha * (reward[prev] + gamma * max(Q[new]) - Q[prev][action])
                # TODO: I think there is some bug in regard to rewarding victories
                if reward[t] == 1:
                    update[prev_states[t]][action] = 1
                    loser = self.q_table_1 if self.current_turn == 'O' else self.q_table_2
                    loser[prev_move[0]][prev_move[1]] = -1
                elif len(action_space) == 1:
                    update[prev_states[t]][action] = 0
                    other = self.q_table_1 if self.current_turn == 'O' else self.q_table_2
                    other[prev_move[0]][prev_move[1]] = 0
                prev_move = (prev_states[t], action)
                prev_states[t] = new_state
                t = (t+1) % 2
            self.reset()
            self.epsilon = np.exp(-decay_rate * episode)

    def one_hot_encode(self):
        size = len(self.board)
        # Creating a 3D array with dimensions [3, size, size]
        # Layer 0: Empty, Layer 1: X, Layer 2: O
        one_hot_encoded = [[[0 for _ in range(size)] for _ in range(size)] for _ in range(3)]

        for row in range(size):
            for col in range(size):
                if self.board[row][col] == '':
                    one_hot_encoded[0][row][col] = 1  # Mark empty cell
                elif self.board[row][col] == 'X':
                    one_hot_encoded[1][row][col] = 1  # Mark X cell
                elif self.board[row][col] == 'O':
                    one_hot_encoded[2][row][col] = 1  # Mark O cell

        return one_hot_encoded

    def state_to_index(self): # Same as observation space ?
        to_num = {'':0, 'X':1, 'O':2}
        num = 0
        for i, row in enumerate(self.board):
            for j, sq in enumerate(row):
                num += np.power(3, self.size*i+j)*to_num[sq]
        return num

    def observation_space_size(self):
        return 3**(self.size**2)

    def action_space_get(self):
        """
        Returns the indices of the empty squares
        :return: list[int]
        """
        actions = []
        for i, row in enumerate(self.board):
            for j, sq in enumerate(row):
                if sq == '':
                    actions.append(self.size*i+j)
        return actions

    def action_space_size(self):
        return self.size**2

    def step(self, action, draw=False):
        """
        Plays the action in the board
        :param action: int
        :return: None
        """
        row = action // self.size
        col = action % self.size
        self.mark_square(row, col, draw)
        return self.state_to_index(), self.reward(), self.game_over

    def reward(self):
        if self.game_over:
            if self.current_turn == 'O':
                return [1, -1]
            else:
                return [-1, 1]
        else:
            return [0, 0]

    def reset(self, view=False):
        self.board = [['' for _ in range(self.size)] for _ in range(self.size)]
        self.current_turn = 'X'
        self.moves_played = 0
        self.game_over = False
        if view:
            self.screen.fill((255, 255, 255))
            self.draw_board()

    def play_against_trained(self, start=True):
        self.epsilon = 0 # never explore
        self.reset(True)
        turn_ai = 1 if start else 0
        if not start:
            self.step(self.sample_actions(self.action_space_get(), turn_ai))

        print(f"Corresponding values\n"
              f"{self.q_table_2[self.state_to_index()][0:3]}"
              f"\n{self.q_table_2[self.state_to_index()][3:6]}"
              f"\n{self.q_table_2[self.state_to_index()][6:9]}")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    mouseX = event.pos[0] // self.cell_size
                    mouseY = event.pos[1] // self.cell_size
                    self.mark_square(mouseY, mouseX, True)

                    if self.game_over:
                        break

                    actions = self.action_space_get()

                    if len(actions) == 0:
                        break

                    print("Actions: ", actions, turn_ai)

                    print(f"Corresponding values\n"
                          f"{self.q_table_1[self.state_to_index()][0:3]}"
                          f"\n{self.q_table_1[self.state_to_index()][3:6]}"
                          f"\n{self.q_table_1[self.state_to_index()][6:9]}")

                    action = self.sample_actions(actions, turn_ai)
                    print("Chosen: ", action)
                    print(f"Current state: {self.state_to_index()}")
                    self.step(action, True)
            pygame.display.update()
        pygame.quit()

    def watch_ai_play(self):
        pass


# To play a 3x3 game where 3 marks in a row are needed to win
game = TicTacToeGUI(3, 3)
game.train(10000)
game.play_against_trained()
#game.run_game()


# TODO: implement play after training CHECK
# TODO: implement watch AI play
# TODO: improve data-structures (use mode numpy?)
# TODO: do something when game ends


