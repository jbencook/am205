import numpy as np

eps = 1e-3

def SIGN(a,b):
	if b >= 0.0:
		return np.abs(a)
	else:
		return -np.abs(a)

def MAX(a,b):
	if a >= b:
		return a
	else:
		return b

def PYTHAG(a,b):
	absa = np.abs(a)
	absb = np.abs(b)
	if absa > absb:
		return absa * np.sqrt(1.0 + np.square(absb/absa))
	else:
		return absb * np.sqrt(1.0 + np.square(absa/absb))

def svd(A):
	#Setup
	m,n = A.shape
	assert m >= n, 'm must be greater than or equal to n'
	U = A.copy()
	w = np.zeros(n, dtype=np.float32)
	V = np.zeros((n,m), dtype=np.float32)

	#Householder reduction to bidirectional form.
	g = scale = anorm = 0.
	rv1 = np.zeros(n, dtype=np.float32)
	for i in xrange(n):
		l = i + 2
		rv1[i] = scale * g
		g = s = scale = 0.
		if i < m:
			scale = np.sum(U[i:,i])
			if scale != 0.0:
				U[i:,i] /= scale
				s = np.linalg.norm(U[i:,i])
				f = U[i,i]
				g = -SIGN(np.sqrt(s),f)
				h = f * g - s
				U[i,i] = f - g
				for j in xrange(l-1,n):
					s = np.sum(U[i:,i]*U[i:,j])
					f = s / h
					U[i:,j] += f * U[i:,i]
				U[i:,i] *= scale
		w[i] = scale * g
		g = s = scale = 0.
		if (i + 1) <= m and (i + 1) != n:
			scale = np.sum(np.abs(U[i,(l-1):]))
			if scale != 0.0:
				U[i,(l-1):] /= scale
				s = np.linalg.norm(U[i,(l-1):])
				f = U[i,l-1]
				g = -SIGN(np.sqrt(s),f)
				h = f * g - s
				U[i,l-1] = f - g
				rv1[(l-1):] = U[i,(l-1):] / h
				for j in xrange(l-1,n):
					s = np.sum(U[j,(l-1):]*U[i,(l-1):])
					U[j,(l-1):] += s * rv1[(l-1):]
				U[i,(l-1):] *= scale
		anorm = MAX(anorm, np.abs(w[i]) + np.abs(rv1[i]))

		#Accumulation of right-hand transformations
		for i in range(n)[::-1]:
			if i < n-1:
				if g != 0.0:
					V[:,i] = (U[i,:] / U[i,l]) / g
					for j in xrange(l,n):
						s = np.sum(U[i,l:]*V[l:,j])
						V[l:,j] = s*V[l:,i]
				V[i,l:] = 0.
				V[l:,i] = 0.
			V[i,i] = 1.
			g = rv1[i]
			l = i

		#Accumulation of left-hand transformations
		for i in range(n)[::-1]:
			l + i + 1
			g = w[i]
			U[i,l:] = 0.
			if g != 0.0:
				g = 1.0 / g
				for j in xrange(l,n):
					s = np.sum(U[l:,i]*U[l:,j])
					f = (s / U[i,i]) * g
					U[i:,j] += f * U[i:,i]
				U[i:,i] *= g
			else:
				U[i:,i] = 0.
			U[i,i] += 1

		#Diagonalization of the bidiagonal form: Loop over
		#singular values, and over allowed iterations.
		for k in range(n)[::-1]:
			for its in xrange(30):
				flag = True
				for l in range(k)[::-1]:
					nm = l-1
					if l == 0 or np.abs(rv1[l]) <= eps*anorm:
						flag = False
						break
					if np.abs(w[nm]) <= eps*anorm:
						break
				if flag:
					c = 0.0			#Cancellation of rv1[l] if l > 0
					s = 1.0
					for i in xrange(l,k+1):
						f = s * rv1[i]
						rv1[i] = c*rv1[i]
						if np.abs(f) <= eps * anorm:
							break
						g = w[i]
						h = PYTHAG(f,g)
						w[i] = h = 1.0 / h
						c = g * h
						s = -f * h
						for j in xrange(m):
							y = U[j,nm]
							z = U[j,i]
							U[j,nm] = y * c + z * s
							U[j,i] = z * c - y * s




if __name__ == '__main__':
	A = np.diag(np.arange(10))
	svd(A)