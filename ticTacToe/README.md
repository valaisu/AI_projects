The goal is to make a program for training a tic-tac-toe AI with Q-learning

The size of the board and lane length required to win can be altered.

The goal is to test the limits of q-learning. The size of the q table grows rapidly as the size of the board is increased. 
An upper limit for the q-table size can be calculated 3^(n^2) * (n^2)/2 which corresponds to (different states of square)^(board_width^2) * (possible actions/turn)
For board size 3, 4, 5, these size are

3 :  1.0*10^4
4 :  2.1*10^7
5 :  4.2*10^11

but obviously all of the empty-X-O combinations are not allowed
