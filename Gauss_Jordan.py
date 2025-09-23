# Funciones auxiliares
def copiar_matriz(A): # Copia profunda de matriz para no alterar el argumento original
    return [fila[:] for fila in A]

def es_cuadrada(A): # Verifica formato n x n
    n = len(A)
    return all(len(f) == n for f in A)

def es_diagonal_dominante(A):
    n = len(A)
    for i in range(n):
        if not (abs(A[i][i]) > sum(abs(A[i][j]) for j in range(n) if j != i)):
            return False
    return True

def producto_matriz_vector(A, x):
    n = len(A)
    r = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j]*x[j]
        r[i] = s
    return r

def vector_finito(x):
    for v in x:
        if v != v or v == float("inf") or v == float("-inf"):
            return False
    return True

def norma_infinito(v):
    m = 0.0
    for val in v:
        if val != val or val == float("inf") or val == float("-inf"):
            return float("inf")
        av = abs(val)
        if av > m: m = av
    return m

def residuo(A, x, b):
    if not vector_finito(x):
        return [float("nan")]*len(b), float("inf")
    Ax = producto_matriz_vector(A, x)
    r = [bi - axi for bi, axi in zip(b, Ax)]
    return r, norma_infinito(r)

def imprimir_vector(nombre, v):
    print(f"{nombre} = [", end="")
    for i, val in enumerate(v):
        s = f"{val:.10f}" if (val == val and abs(val) != float("inf")) else "nan"
        print(s + (", " if i < len(v)-1 else ""), end="")
    print("]")

def casos_prueba():
    A1 = [[10,-1, 2],[-1,11,-1],[2,-1,10]]
    b1 = [6,25,-11]
    A2 = [[1,3,-2],[4,1,-1],[-2,-1,5]]
    b2 = [5,6,4]
    return (A1,b1,"A1x=b1 (DD)"), (A2,b2,"A2x=b2 (No DD)")

# Gauss-Jordán 

def gauss_jordan(A_in, b_in, pivoteo_parcial=True):
    A = copiar_matriz(A_in)
    b = b_in[:]
    n = len(A)

    if not es_cuadrada(A) or len(b) != n:
        raise ValueError("Dimensiones inconsistentes")

    # Recorremos columna k=0..n-1
    for k in range(n):
        # 1) Pivoteo parcial en columna k (elegir fila con máximo |A[i][k]|)
        if pivoteo_parcial:
            p = max(range(k, n), key=lambda i: abs(A[i][k]))
            if abs(A[p][k]) == 0.0:
                raise ValueError("Matriz singular (pivote cero)")
            if p != k:
                A[k], A[p] = A[p], A[k]
                b[k], b[p] = b[p], b[k]
        else:
            if A[k][k] == 0.0:
                raise ValueError("Pivote cero sin pivoteo")

        # 2) Normalizar fila k para que el pivote sea 1
        piv = A[k][k]
        for j in range(n): A[k][j] /= piv
        b[k] /= piv

        # 3) Anular la columna k en TODAS las filas i != k (arriba y abajo)
        for i in range(n):
            if i != k:
                factor = A[i][k]
                for j in range(n): A[i][j] -= factor * A[k][j]
                b[i] -= factor * b[k]

    # Al terminar, A ≈ I y el vector b contiene x
    return b

def main():
    for (A, b, name) in casos_prueba():
        print("="*70)
        print("Gauss-Jordán —", name)
        print("Diagonal dominante:", es_diagonal_dominante(A))
        try:
            x = gauss_jordan(A, b, True)
            imprimir_vector("x", x)
            _, nr = residuo(A, x, b)
            print(f"||residuo||_inf = {nr:.3e}")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()