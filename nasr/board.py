import sys
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from pyswip import Prolog
import numpy as np


def solve_sudoku(input_problem, prolog_instance=None):
    try:
        if not prolog_instance:
            prolog_instance = Prolog()
            prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
        input_type = type(input_problem)
        if input_type == list:
            input_problem = str(input_problem).replace('0','_')
        elif input_type == np.ndarray:
            input_problem = str(input_problem).replace('0','_').replace('\n','').replace(' ',',')
        
        solution_list =  list(prolog_instance.query("Rows=%s,sudoku(Rows)" % input_problem, maxresult=1))
        solution = []
        if len(solution_list)>0:
            solution = solution_list[0]["Rows"]
            if input_type == np.ndarray:
                solution = np.array(solution).astype(int)
        return solution
    except Exception as e:
        print('------------ prolog crashed')
        return []


def expand_line(line):
    base = 3
    return line[0] + line[5:9].join([line[1:5] * (base - 1)] * base) + line[9:13]


def check_input_board(input_board,pred_board):
    input_board = input_board.reshape(81)
    pred_board = pred_board.reshape(81)
    for i in range(81):
        if input_board[i] != 0:
            if input_board[i] != pred_board[i]:
                return
    return True


def check_consistency_board(pred_board):
    board = pred_board.reshape(9,9)
 # Check row
    for k in range(9):
        row = board[k]
        if 9 != len(set(row)):
            return False
        column = [board[j][k] for j in range(9)]
        if 9 != len(set(column)):
            return False
        box = [(3*(k//3),j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                [(3*(k//3)+1,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                [(3*(k//3)+2,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)]
        box = [board[i][j] for (i,j) in box]
        if 9 != len(set(box)):
            return False
    return True


class Board:
    def __init__(self, board_init=None):
        if board_init is None:
            self.board = np.zeros((9, 9), dtype=int)
        else:
            self.board = np.array(board_init)
        self.visual_board = None
    

    def solve(self, solver = 'prolog', prolog_instance = None):
        '''
        @solver : 'prolog', 'backtrack'
        '''
        if self.input_is_valid() == False:
            return False
        if solver == 'prolog':
            if prolog_instance:
                solution = solve_sudoku(self.board, prolog_instance)
            else:
                solution = solve_sudoku(self.board)
            if len(solution)>0:
                self.board = solution
                return True
            else:
                return False
        elif solver == 'backtrack':
            find = self.find_empty()
            if not find:
                return True
            else:
                row, col = find
            for i in range(1,10):
                if self.is_valid(i, (row, col)):
                    self.board[row][col] = i
                    if self.solve('backtrack'):
                        return True
                    self.board[row][col] = 0
            return False

    def board_string(self):
        out = self.board.reshape(81,1).tolist()
        out = [i[0] for i in out]
        out = ''.join(str(i) for i in out)
        return out

    def print_board(self):
        print('\n')
        for i in range(len(self.board)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")
            for j in range(len(self.board[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                if j == 8:
                    print(self.board[i][j])
                else:
                    print(str(self.board[i][j]) + " ", end="")

    def find_empty(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    return i, j  # row, col
        return None

    def is_valid(self, num, pos):
        # Check row
        for i in range(len(self.board[0])):
            if self.board[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(len(self.board)):
            if self.board[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.board[i][j] == num and (i,j) != pos:
                    return False
        return True


    def input_is_valid(self):
        n = len(self.board[0])
        for k in range(n):
            row = self.board[k]
            row = list(filter(lambda a: a != 0, row))
            if len(row) != len(set(row)):
                return False
            column = [self.board[j][k] for j in range(n)]
            column = list(filter(lambda a: a != 0, column))
            if len(column) != len(set(column)):
                return False
            box = [(3*(k//3),j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                    [(3*(k//3)+1,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)] + \
                    [(3*(k//3)+2,j) for j in range(3*(k-3*(k//3)),3*(k-3*(k//3))+3)]
            box = [self.board[i][j] for (i,j) in box]
            box = list(filter(lambda a: a != 0, box))
            if len(box) != len(set(box)):
                return False
        return True      

