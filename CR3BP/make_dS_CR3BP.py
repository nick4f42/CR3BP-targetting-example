import os

import sympy as sp
from sympy.utilities.codegen import C99CodeGen

from make_routine import make_routine

# CR3BP Equations of Motion
S = sp.Matrix(sp.symbols('S:6', real=True))
mu = sp.Symbol('mu', positive=True)

r13 = S[:3, :] - sp.Matrix([-mu, 0, 0])
r23 = S[:3, :] - sp.Matrix([1 - mu, 0, 0])

accel = (sp.Matrix([2 * S[4] + S[0], -2 * S[3] + S[1], 0])
         - (1 - mu) / r13.norm()**3 * r13
         - mu / r23.norm()**3 * r23)

# Derivative of state vector
dS = sp.Matrix([*S[3:], *accel])

dS_mat = sp.MatrixSymbol('dS', 6, 1)
S_mat = sp.MatrixSymbol('S', 6, 1)

matrix_map = dict(zip(S_mat, S))
dS_eq = sp.Eq(dS_mat, dS.xreplace(matrix_map))

# Derivative of state transition matrix
STM_cols = 3
STM = sp.Matrix(sp.symbols(f'STM:{6 * STM_cols}', real=True)).reshape(6, STM_cols)
dSTM = dS.jacobian(S) @ STM

STM_mat = sp.MatrixSymbol('STM', 6, STM_cols)
dSTM_mat = sp.MatrixSymbol('dSTM', 6, STM_cols)

matrix_map.update(dict(zip(STM_mat, STM)))
dSTM_eq = sp.Eq(dSTM_mat, dSTM.xreplace(matrix_map))

# Generate c files

if __name__ == '__main__':
    gen = C99CodeGen()

    routines = [
        make_routine(
            'c_dS_CR3BP',
            [S_mat, mu],
            [dS_eq],
            var_map=matrix_map
        ),
        make_routine(
            'c_dSTM_CR3BP',
            [S_mat, STM_mat, mu],
            [dS_eq, dSTM_eq],
            var_map=matrix_map
        )
    ]

    dirpath = os.path.dirname(os.path.abspath(__file__))
    c_path = os.path.join(dirpath, 'c_CR3BP')

    gen.write(
        routines,
        c_path,
        to_files=True,
        header=False,
        empty=False
    )
