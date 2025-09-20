# Resolucion_Sistemas_De_Eecuaciones_Lineales_Mediante_Metodos_Num-ricos_En_Python

Este repositorio contiene la implementación en **Python** (sin librerías externas) de distintos métodos para resolver sistemas de ecuaciones lineales:

- Eliminación de **Gauss** (método directo, con pivoteo parcial)
- **Gauss-Jordán** (método directo)
- **Jacobi** (método iterativo)
- **Gauss-Seidel** (método iterativo)

## Características
- Uso de estructuras básicas de Python (listas, ciclos, condicionales).
- Pruebas con:
  - Una matriz **diagonalmente dominante** (los métodos iterativos convergen).
  - Una matriz **no diagonalmente dominante** (los iterativos pueden no converger).
- Verificación de la solución mediante el **cálculo del residuo**.
- Código documentado con comentarios explicativos.

## Ejecución
```bash
python Eliminacion_Gauss.py
python Gauss_Jordan.py
python Jacobi.py
python Gauss_Seidel.py
