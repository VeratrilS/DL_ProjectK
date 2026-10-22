"""Lab 0: Linear Algebra & Matrix Foundations with NumPy.

Used in Sketch2React for:
- Rotating and straightening tilted hand-drawn sketches
- Verifying transformation matrices are strictly ORTHOGONAL (no shape shearing)
- Computing image stroke dispersion via Matrix TRACE
- Ensuring non-singular affine mappings via DETERMINANT & GAUSSIAN ELIMINATION
"""

import numpy as np


def verify_orthogonality(matrix: np.ndarray) -> bool:
  """Verifies if a square matrix A is Orthogonal: A .

  A^T = I
  """
  n = matrix.shape[0]
  transpose = matrix.T
  product = np.dot(matrix, transpose)
  identity = np.eye(n)
  return np.allclose(product, identity)


def compute_trace(matrix: np.ndarray) -> float:
  """Computes the Trace: sum of the main diagonal elements."""
  return float(np.trace(matrix))


def determinant_via_gaussian_elimination(matrix: np.ndarray) -> float:
  """Computes the determinant by row-reducing matrix to Upper Triangular (U).

  Determinant is the product of diagonal elements: det(A) = prod(U_ii)
  """
  U = matrix.astype(float).copy()
  n = len(U)
  det_sign = 1.0

  for i in range(n):
    # Pivot selection
    if U[i][i] == 0:
      # Swap with non-zero row below
      swapped = False
      for k in range(i + 1, n):
        if U[k][i] != 0:
          U[[i, k]] = U[[k, i]]
          det_sign *= -1.0  # Row swap flips determinant sign
          swapped = True
          break
      if not swapped:
        return 0.0  # Singular matrix

    # Row reduction
    for j in range(i + 1, n):
      factor = U[j][i] / U[i][i]
      U[j] -= factor * U[i]

  det = det_sign * float(np.prod([U[i][i] for i in range(n)]))
  return det


def create_orthogonal_rotation_matrix(angle_degrees: float) -> np.ndarray:
  """Creates a 2D rotation matrix for sketch orientation correction and asserts orthogonality."""
  rad = np.radians(angle_degrees)
  R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
  assert verify_orthogonality(R), (
      "Error: Rotation matrix must be strictly orthogonal!"
  )
  return R


# --- Interactive Lab Revision Demo ---
if __name__ == "__main__":
  print("=" * 60)
  print("LAB 0: NUMPY LINEAR ALGEBRA FOUNDATIONS")
  print("=" * 60)

  # 1. Orthogonality test
  R_15 = create_orthogonal_rotation_matrix(15.0)
  print("\n1. 2D Sketch Rotation Matrix (15 degrees):\n", R_15)
  print("Is Orthogonal (R . R^T == I)?", verify_orthogonality(R_15))

  # 2. Gaussian elimination determinant test
  test_mat = np.array([[3.0, 2.0, 1.0], [1.0, 4.0, 2.0], [2.0, 1.0, 5.0]])
  det_gauss = determinant_via_gaussian_elimination(test_mat)
  det_numpy = np.linalg.det(test_mat)
  print("\n2. Determinant Calculation:")
  print(f"Gaussian Elimination det: {det_gauss:.4f}")
  print(f"NumPy linalg det:         {det_numpy:.4f}")

  # 3. Trace and Invertibility
  print(f"\n3. Trace (Diagonal Sum): {compute_trace(test_mat):.4f}")
  if det_gauss != 0:
    print("Matrix is Invertible. Inverse:\n", np.linalg.inv(test_mat))
  print("=" * 60)
