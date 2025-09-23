#Funciones auxiliares
def copiar_matriz(A): # Devuelve una copia independiente de A (para no mutar el argumento)
    return [fila[:] for fila in A]

def es_cuadrada(A): # Verifica formato n x n (requisito para sistemas lineales A x = b)
    n = len(A)
    return all(len(fila) == n for fila in A)

def es_diagonal_dominante(A): # Criterio útil para iterativos 
    # |a_ii| > sum_{j != i} |a_ij| para todas las filas
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        suma = sum(abs(A[i][j]) for j in range(n) if j != i)
        if not (diag > suma):
            return False
    return True

def producto_matriz_vector(A, x): # Implementación manual de Ax (sin librerías)
    n = len(A)
    res = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j]*x[j]
        res[i] = s
    return res

def vector_finito(x): # True si todos los componentes son números finitos (no NaN/inf)
    for v in x:
        if v != v or v == float("inf") or v == float("-inf"):
            return False
    return True

def norma_infinito(v): # max |v_i|; si hay NaN/inf, se considera infinita
    m = 0.0
    for val in v:
        if val != val or val == float("inf") or val == float("-inf"):
            return float("inf")
        av = abs(val)
        if av > m:
            m = av
    return m

def residuo(A, x, b): # r = b - A x y su norma; si x no es finito, reporta infinito
    if not vector_finito(x):
        return [float("nan")]*len(b), float("inf")
    Ax = producto_matriz_vector(A, x)
    r = [bi - axi for bi, axi in zip(b, Ax)]
    return r, norma_infinito(r)

def imprimir_vector(nombre, v): # Imprime vector con formato y maneja NaN/inf para evitar fallos al formatear
    print(f"{nombre} = [", end="")
    for i, val in enumerate(v):
        s = f"{val:.10f}" if (val == val and abs(val) != float("inf")) else "nan"
        print(s + (", " if i < len(v)-1 else ""), end="")
    print("]")

def casos_prueba(): 
    # Sistema DD (apto para iterativos, directos siempre funcionan si A no es singular)
    A1 = [[10.0, -1.0,  2.0],
          [ -1.0, 11.0, -1.0],
          [  2.0, -1.0, 10.0]]
    b1 = [6.0, 25.0, -11.0]
    # Sistema No DD (iterativos pueden no converger; directos sí si A es no singular)
    A2 = [[ 1.0,  3.0, -2.0],
          [ 4.0,  1.0, -1.0],
          [-2.0, -1.0,  5.0]]
    b2 = [5.0, 6.0, 4.0]
    return (A1, b1, "A1x=b1 (DD)"), (A2, b2, "A2x=b2 (No DD)")

#  Algoritmo de Gauss con pivoteo parcial 
def eliminacion_gauss(A_in, b_in, pivoteo_parcial=True):
    # Trabaja sobre COPIAS para no modificar A_in y b_in
    A = copiar_matriz(A_in)
    b = b_in[:]
    n = len(A)

    # Chequeo de consistencia de dimensiones
    if not es_cuadrada(A) or len(b) != n:
        raise ValueError("Dimensiones inconsistentes")
    
    # FASE DE ELIMINACIÓN: hacemos ceros por debajo de la diagonal
    for k in range(n-1): # --- Pivoteo parcial: buscamos el mayor |A[i][k]| en i=k..n-1 ---
        if pivoteo_parcial:
            p = max(range(k, n), key=lambda i: abs(A[i][k]))
            if abs(A[p][k]) == 0.0: # Si el mejor pivote posible es 0, la matriz es singular
                raise ValueError("Matriz singular (pivote cero)")
            if p != k: # Intercambiamos filas en A y b
                A[k], A[p] = A[p], A[k]
                b[k], b[p] = b[p], b[k]
        else: # Sin pivoteo, debemos garantizar A[k][k] != 0
            if A[k][k] == 0.0:
                raise ValueError("Pivote cero sin pivoteo")

        # Eliminación: para cada fila i>k, anulamos A[i][k]
        for i in range(k+1, n):
            m = A[i][k] / A[k][k]  # multiplicador
            # R_i <- R_i - m * R_k (desde la columna k para ahorrar operaciones)
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]

    # Chequeo del último pivote
    if A[n-1][n-1] == 0.0:
        raise ValueError("Matriz singular en último pivote")

    # SUSTITUCIÓN HACIA ATRÁS: resolvemos el sistema triangular superior
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        # s = b_i - sum(a_ij * x_j) con j > i (los x_j ya se conocen)
        s = b[i]
        for j in range(i+1, n):
            s -= A[i][j]*x[j]
        x[i] = s / A[i][i]
    return x

def main():
    # Ejecutamos los DOS sistemas (DD y No DD) para cumplir el requisito de pruebas
    for (A, b, name) in casos_prueba():
        print("="*70)
        print("Gauss —", name)
        print("Diagonal dominante:", es_diagonal_dominante(A))
        try:
            x = eliminacion_gauss(A, b, pivoteo_parcial=True)
            imprimir_vector("x", x)
            _, nr = residuo(A, x, b)
            print(f"||residuo||_inf = {nr:.3e}")  # verificación de solución
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
