import random

# Evaluación Heurística del tablero
def evaluate_board(board, player):
    opponent = -player
    player_score = 0
    opponent_score = 0
    player_mobility = 0
    opponent_mobility = 0
    player_corners = 0
    opponent_corners = 0
    player_frontier = 0
    opponent_frontier = 0

    # Asignación de pesos a las posiciones del tablero
    weights = [
        [20, -3, 11, 8, 8, 11, -3, 20],
        [-3, -7, -4, 1, 1, -4, -7, -3],
        [11, -4, 2, 2, 2, 2, -4, 11],
        [8, 1, 2, -3, -3, 2, 1, 8],
        [8, 1, 2, -3, -3, 2, 1, 8],
        [11, -4, 2, 2, 2, 2, -4, 11],
        [-3, -7, -4, 1, 1, -4, -7, -3],
        [20, -3, 11, 8, 8, 11, -3, 20]
    ]

    # Calcular la movilidad, las esquinas y la frontera
    for row in range(8):
        for col in range(8):
            if board[row][col] == player:
                player_score += weights[row][col]
                if is_valid_move(board, player, row, col):
                    player_mobility += 1
                if (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    player_corners += 1
                if has_opponent_neighbor(board, row, col, opponent):
                    player_frontier += 1
            elif board[row][col] == opponent:
                opponent_score += weights[row][col]
                if is_valid_move(board, opponent, row, col):
                    opponent_mobility += 1
                if (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    opponent_corners += 1
                if has_opponent_neighbor(board, row, col, player):
                    opponent_frontier += 1

    # Puntuación final con movilidad, esquinas y frontera
    final_score = player_score - opponent_score + (player_mobility - opponent_mobility) * 10 + (player_corners - opponent_corners) * 20 + (player_frontier - opponent_frontier) * 5

    return final_score

def has_opponent_neighbor(board, row, col, opponent):
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in neighbors:
        r, c = row + dr, col + dc
        if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
            return True
    return False

# Verifica si un movimiento es válido
def is_valid_move(board, player, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    if board[row][col] != 0:
        return False

    opponent = -player

    for direction in directions:
        dr, dc = direction
        r, c = row + dr, col + dc
        found_opponent = False

        while 0 <= r < 8 and 0 <= c < 8 and board[r][c] == opponent:
            r += dr
            c += dc
            found_opponent = True

        if found_opponent and 0 <= r < 8 and 0 <= c < 8 and board[r][c] == player:
            return True

    return False

# Obtiene los movimientos válidos
def valid_moves(board, player):
    valid_moves = []
    for row in range(8):
        for col in range(8):
            if is_valid_move(board, player, row, col):
                valid_moves.append((row, col))

    return valid_moves

# Realiza un movimiento en el tablero y devuelve el nuevo estado del tablero
def make_move(board, player, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    new_board = [row[:] for row in board]
    new_board[row][col] = player

    for direction in directions:
        dr, dc = direction
        r, c = row + dr, col + dc
        pieces_to_flip = []

        while 0 <= r < 8 and 0 <= c < 8 and new_board[r][c] == -player:
            pieces_to_flip.append((r, c))
            r += dr
            c += dc

        if 0 <= r < 8 and 0 <= c < 8 and new_board[r][c] == player:
            for rr, cc in pieces_to_flip:
                new_board[rr][cc] = player

    return new_board

# Implementa el algoritmo Minimax con poda Alpha-Beta y tablas de transposición
transposition_table = {}

def minimax(board, player, depth, alpha, beta, maximizing):
    # Verificar si la posición actual está en la tabla de transposición
    board_key = tuple(map(tuple, board))
    if board_key in transposition_table:
        return transposition_table[board_key]

    valid_moves_list = valid_moves(board, player)

    if depth == 0 or not valid_moves_list:
        return evaluate_board(board, player), None

    best_move = None

    if maximizing:
        max_eval = float('-inf')
        # Ordenar los movimientos en orden descendente de puntuación heurística
        ordered_moves = sorted(valid_moves_list, key=lambda m: evaluate_board(make_move(board, player, m[0], m[1]), player), reverse=True)
        for move in ordered_moves:
            new_board = make_move(board, player, move[0], move[1])
            eval, _ = minimax(new_board, -player, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[board_key] = max_eval, best_move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        # Ordenar los movimientos en orden ascendente de puntuación heurística
        ordered_moves = sorted(valid_moves_list, key=lambda m: evaluate_board(make_move(board, player, m[0], m[1]), player))
        for move in ordered_moves:
            new_board = make_move(board, player, move[0], move[1])
            eval, _ = minimax(new_board, -player, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[board_key] = min_eval, best_move
        return min_eval, best_move

# Función AI_MOVE mejorada con movimientos de apertura codificados
opening_book = {
    # Ejemplo de movimientos de apertura codificados
    ((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0), 1): (3, 3),
    ((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0), -1): (3, 4),
    # Se pueden agregar más movimientos de apertura codificados aquí
}

def AI_MOVE(board, player):
    board_tuple = tuple(map(tuple, board))
    if (board_tuple, player) in opening_book:
        return opening_book[(board_tuple, player)]

    depth = 6  # Profundidad del árbol de búsqueda
    _, best_move = minimax(board, player, depth, float('-inf'), float('inf'), True)
    return best_move