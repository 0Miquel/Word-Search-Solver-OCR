import numpy as np


def word_search_solver(matrix, word):  # de moment només horitzontal i vertical cap a la dreta i abaix
    """

    :param matrix, word:
    :return:
    """
    trobat = False
    word = [ch for ch in word]
    positions = []
    w = word
    letter = w[0]
    w = w[1:]

    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if not trobat:
                trobat = word_search_solver_recursive_horitzontal(matrix, w, letter, row, col,
                                                                  positions)  # word te at, letter te c
                if (trobat == False and positions != []):
                    positions = []
                    letter = word[0]
                    w = word
                    letter = w[0]
                    w = w[1:]
            else:
                break
    if not trobat:
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if not trobat:
                    trobat = word_search_solver_recursive_vertical(matrix, w, letter, row, col,
                                                                   positions)  # word te at, letter te c
                    if (trobat == False and positions != []):
                        positions = []
                        letter = word[0]
                        w = word
                        letter = w[0]
                        w = w[1:]
                else:
                    break
    if not trobat:
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if not trobat:
                    trobat = word_search_solver_recursive_diagonal_dreta(matrix, w, letter, row, col,
                                                                         positions)  # word te at, letter te c
                    if (trobat == False and positions != []):
                        positions = []
                        letter = word[0]
                        w = word
                        letter = w[0]
                        w = w[1:]
                else:
                    break
    if not trobat:
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if not trobat:
                    trobat = word_search_solver_recursive_diagonal_esquerra(matrix, w, letter, row, col,
                                                                            positions)  # word te at, letter te c
                    if (trobat == False and positions != []):
                        positions = []
                        letter = word[0]
                        w = word
                        letter = w[0]
                        w = w[1:]
                else:
                    break

    return positions, trobat


def word_search_solver_recursive_horitzontal(matrix, word, letter, row, col, positions):
    if word == []:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                return True
            else:
                return False
        else:
            return False
    else:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                letter = word[0]
                word = word[1:]
                return word_search_solver_recursive_horitzontal(matrix, word, letter, row, col + 1,
                                                                positions)  # horitzontal
            else:
                return False
        else:
            return False


def word_search_solver_recursive_vertical(matrix, word, letter, row, col, positions):
    if word == []:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                return True
            else:
                return False
        else:
            return False
    else:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                letter = word[0]
                word = word[1:]
                return word_search_solver_recursive_vertical(matrix, word, letter, row + 1, col,
                                                             positions)  # horitzontal
            else:
                return False
        else:
            return False


def word_search_solver_recursive_diagonal_dreta(matrix, word, letter, row, col, positions):
    if word == []:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                return True
            else:
                return False
        else:
            return False
    else:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                letter = word[0]
                word = word[1:]
                return word_search_solver_recursive_diagonal_dreta(matrix, word, letter, row + 1, col + 1, positions)
            else:
                return False
        else:
            return False


def word_search_solver_recursive_diagonal_esquerra(matrix, word, letter, row, col, positions):
    if word == []:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                return True
            else:
                return False
        else:
            return False
    else:
        if row < np.shape(matrix)[0] and col < np.shape(matrix)[1]:
            if matrix[row][col] == letter:
                positions.append((row, col))
                letter = word[0]
                word = word[1:]
                return word_search_solver_recursive_diagonal_esquerra(matrix, word, letter, row + 1, col - 1, positions)
            else:
                return False
        else:
            return False

