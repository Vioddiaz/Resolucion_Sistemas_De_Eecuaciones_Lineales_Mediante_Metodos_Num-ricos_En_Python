# Funciones auxiliares 
def copiar_matriz(A):
    return [fila[:] for fila in A]

def es_cuadrada(A):
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

# ========== Gauss-Seidel ==========

def gauss_seidel(A, b, x0=None, tol=1e-10, maxiter=5000):
    n = len(A)
    # Vector inicial (por defecto ceros)
    x = [0.0]*n if x0 is None else x0[:]

    for it in range(1, maxiter+1):
        x_old = x[:]  # guardamos para medir el cambio de la iteración

        for i in range(n):
            # s1 suma con valores NUEVOS (ya actualizados en esta iteración)
            s1 = 0.0
            for j in range(i):
                s1 += A[i][j]*x[j]
            # s2 suma con valores VIEJOS (aún no actualizados en esta iteración)
            s2 = 0.0
            for j in range(i+1, n):
                s2 += A[i][j]*x[j]

            if A[i][i] == 0.0:
                # Debemos reordenar filas/columnas si hay ceros en la diagonal
                raise ValueError("Cero en diagonal; reordenar filas/columnas")

            # Actualización inmediata (diferencia clave vs Jacobi)
            x[i] = (b[i] - s1 - s2) / A[i][i]

        # Si hay NaN/inf, no convergió
        if not vector_finito(x):
            return x, it, False

        # Criterio de paro: cambio máximo entre x^(k+1) y x^(k)
        diff = [x[i] - x_old[i] for i in range(n)]
        if norma_infinito(diff) < tol:
            return x, it, True

    # No se alcanzó la tolerancia dentro del máximo permitido
    return x, maxiter, False

def main():
    for (A, b, name) in casos_prueba():
        print("="*70)
        print("Gauss-Seidel —", name)
        print("Diagonal dominante:", es_diagonal_dominante(A))
        try:
            x, it, ok = gauss_seidel(A, b, tol=1e-10, maxiter=5000)
            imprimir_vector("x", x)
            print(f"Iteraciones = {it}, Convergió: {ok}")
            _, nr = residuo(A, x, b)
            print(f"||residuo||_inf = {nr:.3e}")
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
