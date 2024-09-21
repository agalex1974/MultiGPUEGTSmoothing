//
// Created by agalex on 7/19/23.
//

#ifndef KNNCUDA_EIGENSOLVER_CUH
#define KNNCUDA_EIGENSOLVER_CUH

template <typename T> __device__ void inline swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}

template <typename T> __device__ void inline copy_vec_elements(T (&a)[3], T (&b)[3])
{
    for (int i = 0; i < 3; i++){
        a[i] = b[i];
    }
}

template <typename T> __device__ void inline copy_arr_elements(T (&a)[3][3], T (&b)[3][3])
{
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            a[i][j] = b[i][j];
        }
    }
}

template <typename T>
class SortEigenstuff
{
public:
    __device__
    void operator()(int sortType, bool isRotation,
                    T (&eval)[3], T (&evec)[3][3])
    {
        if (sortType != 0)
        {
            // Sort the eigenvalues to eval[0] <= eval[1] <= eval[2].
            int index[3];
            if (eval[0] < eval[1])
            {
                if (eval[2] < eval[0])
                {
                    // even permutation
                    index[0] = 2;
                    index[1] = 0;
                    index[2] = 1;
                }
                else if (eval[2] < eval[1])
                {
                    // odd permutation
                    index[0] = 0;
                    index[1] = 2;
                    index[2] = 1;
                    isRotation = !isRotation;
                }
                else
                {
                    // even permutation
                    index[0] = 0;
                    index[1] = 1;
                    index[2] = 2;
                }
            }
            else
            {
                if (eval[2] < eval[1])
                {
                    // odd permutation
                    index[0] = 2;
                    index[1] = 1;
                    index[2] = 0;
                    isRotation = !isRotation;
                }
                else if (eval[2] < eval[0])
                {
                    // even permutation
                    index[0] = 1;
                    index[1] = 2;
                    index[2] = 0;
                }
                else
                {
                    // odd permutation
                    index[0] = 1;
                    index[1] = 0;
                    index[2] = 2;
                    isRotation = !isRotation;
                }
            }

            if (sortType == -1)
            {
                // The request is for eval[0] >= eval[1] >= eval[2]. This
                // requires an odd permutation, (i0,i1,i2) -> (i2,i1,i0).
                swap<int>(index[0], index[2]);
                isRotation = !isRotation;
            }

            T unorderedEVal[3];
            copy_vec_elements<T>(unorderedEVal, eval);
            T unorderedEVec[3][3];
            copy_arr_elements(unorderedEVec, evec);
            for (size_t j = 0; j < 3; ++j)
            {
                size_t i = index[j];
                eval[j] = unorderedEVal[i];
                copy_vec_elements(evec[j], unorderedEVec[i]);
            }
        }

        // Ensure the ordered eigenvectors form a right-handed basis.
        if (!isRotation)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                evec[2][j] = -evec[2][j];
            }
        }
    }
};

template <typename T>
class SymmetricEigensolver3x3
{
public:
    // The input matrix must be symmetric, so only the unique elements
    // must be specified: a00, a01, a02, a11, a12, and a22.
    //
    // If 'aggressive' is 'true', the iterations occur until a
    // superdiagonal entry is exactly zero.  If 'aggressive' is 'false',
    // the iterations occur until a superdiagonal entry is effectively
    // zero compared to the/ sum of magnitudes of its diagonal neighbors.
    // Generally, the nonaggressive convergence is acceptable.
    //
    // The order of the eigenvalues is specified by sortType:
    // -1 (decreasing), 0 (no sorting) or +1 (increasing).  When sorted,
    // the eigenvectors are ordered accordingly, and
    // {evec[0], evec[1], evec[2]} is guaranteed to/ be a right-handed
    // orthonormal set.  The return value is the number of iterations
    // used by the algorithm.
    __device__
    int operator()(T const& a00, T const& a01, T const& a02, T const& a11,
                       T const& a12, T const& a22, bool aggressive, int sortType,
                       T (&eval)[3], T (&evec)[3][3]) const
    {
        // Compute the Householder reflection H0 and B = H0*A*H0, where
        // b02 = 0. H0 = {{c,s,0},{s,-c,0},{0,0,1}} with each inner
        // triple a row of H0.
        T const zero = (T)(0);
        T const one = (T)(1);
        T const half = (T)(0.5);
        bool isRotation = false;
        T c = zero, s = zero;
        GetCosSin(a12, -a02, c, s);
        T term0 = c * a00 + s * a01;
        T term1 = c * a01 + s * a11;
        T term2 = s * a00 - c * a01;
        T term3 = s * a01 - c * a11;
        T b00 = c * term0 + s * term1;
        T b01 = s * term0 - c * term1;
        //T b02 = c * a02 + s * a12;  // 0
        T b11 = s * term2 - c * term3;
        T b12 = s * a02 - c * a12;
        T b22 = a22;

        // Maintain Q as the product of the reflections. Initially,
        // Q = H0. Updates by Givens reflections G are Q <- Q * G. The
        // columns of the final Q are the estimates for the eigenvectors.
        T Q[3][3] = {
                              { c, s, zero },
                              { s, -c, zero },
                              { zero, zero, one }
                       };



        // The smallest subnormal number is 2^{-alpha}. The value alpha is
        // 149 for 'float' and 1074 for 'double'.
        int alpha = 149;
        int i = 0, imax = 0, power = 0;
        T c2 = zero, s2 = zero;

        if (fabsf(b12) <= fabsf(b01))
        {
            // It is known that |currentB12| < 2^{-i/2} * |initialB12|.
            // Compute imax so that 0 is the closest floating-point number
            // to 2^{-imax/2} * |initialB12|.
            frexpf(b12, &power);
            imax = 2 * (power + alpha + 1);

            for (i = 0; i < imax; ++i)
            {
                // Compute the Givens reflection
                // G = {{c,0,-s},{s,0,c},{0,1,0}} where each inner triple
                // is a row of G.
                GetCosSin(half * (b00 - b11), b01, c2, s2);
                s = sqrtf(half * (one - c2));
                c = half * s2 / s;

                // Update Q <- Q * G.
                for (size_t r = 0; r < 3; ++r)
                {
                    term0 = c * Q[r][0] + s * Q[r][1];
                    term1 = Q[r][2];
                    term2 = c * Q[r][1] - s * Q[r][0];
                    Q[r][0] = term0;
                    Q[r][1] = term1;
                    Q[r][2] = term2;
                }
                isRotation = !isRotation;

                // Update B <- Q^T * B * Q, ensuring that b02 is zero and
                // |b12| has strictly decreased.
                term0 = c * b00 + s * b01;
                term1 = c * b01 + s * b11;
                term2 = s * b00 - c * b01;
                term3 = s * b01 - c * b11;
                //b02 = s * c * (b11 - b00) + (c * c - s * s) * b01; // 0
                b00 = c * term0 + s * term1;
                b01 = s * b12;
                b11 = b22;
                b12 = c * b12;
                b22 = s * term2 - c * term3;

                if (Converged(aggressive, b00, b11, b01))
                {
                    // Compute the Householder reflection
                    // H1 = {{c,s,0},{s,-c,0},{0,0,1}} where each inner
                    // triple is a row of H1.
                    GetCosSin(half * (b00 - b11), b01, c2, s2);
                    s = sqrtf(half * (one - c2));
                    c = half * s2 / s;

                    // Update Q <- Q * H1.
                    for (size_t r = 0; r < 3; ++r)
                    {
                        term0 = c * Q[r][0] + s * Q[r][1];
                        term1 = s * Q[r][0] - c * Q[r][1];
                        Q[r][0] = term0;
                        Q[r][1] = term1;
                    }
                    isRotation = !isRotation;

                    // Compute the diagonal estimate D = Q^T * B * Q.
                    term0 = c * b00 + s * b01;
                    term1 = c * b01 + s * b11;
                    term2 = s * b00 - c * b01;
                    term3 = s * b01 - c * b11;
                    b00 = c * term0 + s * term1;
                    b11 = s * term2 - c * term3;
                    break;
                }
            }
        }
        else
        {
            // It is known that |currentB01| < 2^{-i/2} * |initialB01|.
            // Compute imax so that 0 is the closest floating-point number
            // to 2^{-imax/2} * |initialB01|.
            frexpf(b01, &power);
            imax = 2 * (power + alpha + 1);

            for (i = 0; i < imax; ++i)
            {
                // Compute the Givens reflection
                // G = {{0,1,0},{c,0,-s},{s,0,c}} where each inner triple
                // is a row of G.
                GetCosSin(half * (b11 - b22), b12, c2, s2);
                s = sqrtf(half * (one - c2));
                c = half * s2 / s;

                // Update Q <- Q * G.
                for (size_t r = 0; r < 3; ++r)
                {
                    term0 = c * Q[r][1] + s * Q[r][2];
                    term1 = Q[r][0];
                    term2 = c * Q[r][2] - s * Q[r][1];
                    Q[r][0] = term0;
                    Q[r][1] = term1;
                    Q[r][2] = term2;
                }
                isRotation = !isRotation;

                // Update B <- Q^T * B * Q, ensuring that b02 is zero and
                // |b01| has strictly decreased.
                term0 = c * b11 + s * b12;
                term1 = c * b12 + s * b22;
                term2 = s * b11 - c * b12;
                term3 = s * b12 - c * b22;
                //b02 = s * c * (b22 - b11) + (c * c - s * s) * b12;  // 0
                b22 = s * term2 - c * term3;
                b12 = -s * b01;
                b11 = b00;
                b01 = c * b01;
                b00 = c * term0 + s * term1;

                if (Converged(aggressive, b11, b22, b12))
                {
                    // Compute the Householder reflection
                    // H1 = {{1,0,0},{0,c,s},{0,s,-c}} where each inner
                    // triple is a row of H1.
                    GetCosSin(half * (b11 - b22), b12, c2, s2);
                    s = sqrtf(half * (one - c2));
                    c = half * s2 / s;

                    // Update Q <- Q * H1.
                    for (size_t r = 0; r < 3; ++r)
                    {
                        term0 = c * Q[r][1] + s * Q[r][2];
                        term1 = s * Q[r][1] - c * Q[r][2];
                        Q[r][1] = term0;
                        Q[r][2] = term1;
                    }
                    isRotation = !isRotation;

                    // Compute the diagonal estimate D = Q^T * B * Q.
                    term0 = c * b11 + s * b12;
                    term1 = c * b12 + s * b22;
                    term2 = s * b11 - c * b12;
                    term3 = s * b12 - c * b22;
                    b11 = c * term0 + s * term1;
                    b22 = s * term2 - c * term3;
                    break;
                }
            }
        }

        eval[0] = b00;
        eval[1] = b11;
        eval[2] = b22;
        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t col = 0; col < 3; ++col)
            {
                evec[row][col] = Q[col][row];
            }
        }

        SortEigenstuff<T>()(sortType, isRotation, eval, evec);
        return i;
    }

private:
    // Normalize (u,v) to (c,s) with c <= 0 when (u,v) is not (0,0).
    // If (u,v) = (0,0), the function returns (c,s) = (-1,0). When used
    // to generate a Householder reflection, it does not matter whether
    // (c,s) or (-c,-s) is returned. When generating a Givens reflection,
    // c = cos(2*theta) and s = sin(2*theta). Having a negative cosine
    // for the double-angle term ensures that the single-angle terms
    // c = cos(theta) and s = sin(theta) satisfy |c| < 1/sqrt(2) < |s|.
    __device__
    void GetCosSin(T const& u, T const& v, T& c, T& s) const
    {
        T const zero = (T)(0);
        T length = sqrtf(u * u + v * v);
        if (length > zero)
        {
            c = u / length;
            s = v / length;
            if (c > zero)
            {
                c = -c;
                s = -s;
            }
        }
        else
        {
            c = (T)(-1);
            s = zero;
        }
    }

    __device__
    bool Converged(bool aggressive, T const& diagonal0,
                          T const& diagonal1, T const& superdiagonal) const
    {
        if (aggressive)
        {
            // Test whether the superdiagonal term is zero.
            return superdiagonal == (T)(0);
        }
        else
        {
            // Test whether the superdiagonal term is effectively zero
            // compared to its diagonal neighbors.
            T sum = fabsf(diagonal0) + fabsf(diagonal1);
            return sum + fabsf(superdiagonal) == sum;
        }
    }
};


template <typename T>
class NISymmetricEigensolver3x3
{
public:
    // The input matrix must be symmetric, so only the unique elements
    // must be specified: a00, a01, a02, a11, a12, and a22.  The
    // eigenvalues are sorted in ascending order: eval0 <= eval1 <= eval2.

    __device__
    void operator()(T a00, T a01, T a02, T a11, T a12, T a22,
                    int sortType, T (&eval)[3], T (&evec)[3][3])
    {
        // Precondition the matrix by factoring out the maximum absolute
        // value of the components.  This guards against floating-point
        // overflow when computing the eigenvalues.
        T max0 = fmaxf(fabsf(a00), fabsf(a01));
        T max1 = fmaxf(fabsf(a02), fabsf(a11));
        T max2 = fmaxf(fabsf(a12), fabsf(a22));
        T maxAbsElement = fmaxf(fmaxf(max0, max1), max2);
        if (maxAbsElement == (T)0)
        {
            // A is the zero matrix.
            eval[0] = (T)0;
            eval[1] = (T)0;
            eval[2] = (T)0;
            evec[0] = { (T)1, (T)0, (T)0 };
            evec[1] = { (T)0, (T)1, (T)0 };
            evec[2] = { (T)0, (T)0, (T)1 };
            return;
        }

        T invMaxAbsElement = (T)1 / maxAbsElement;
        a00 *= invMaxAbsElement;
        a01 *= invMaxAbsElement;
        a02 *= invMaxAbsElement;
        a11 *= invMaxAbsElement;
        a12 *= invMaxAbsElement;
        a22 *= invMaxAbsElement;

        T norm = a01 * a01 + a02 * a02 + a12 * a12;
        if (norm > (T)0)
        {
            // Compute the eigenvalues of A.

            // In the PDF mentioned previously, B = (A - q*I)/p, where
            // q = tr(A)/3 with tr(A) the trace of A (sum of the diagonal
            // entries of A) and where p = sqrt(tr((A - q*I)^2)/6).
            T q = (a00 + a11 + a22) / (T)3;

            // The matrix A - q*I is represented by the following, where
            // b00, b11 and b22 are computed after these comments,
            //   +-           -+
            //   | b00 a01 a02 |
            //   | a01 b11 a12 |
            //   | a02 a12 b22 |
            //   +-           -+
            T b00 = a00 - q;
            T b11 = a11 - q;
            T b22 = a22 - q;

            // The is the variable p mentioned in the PDF.
            T p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + norm * (T)2) / (T)6);

            // We need det(B) = det((A - q*I)/p) = det(A - q*I)/p^3.  The
            // value det(A - q*I) is computed using a cofactor expansion
            // by the first row of A - q*I.  The cofactors are c00, c01
            // and c02 and the determinant is b00*c00 - a01*c01 + a02*c02.
            // The det(B) is then computed finally by the division
            // with p^3.
            T c00 = b11 * b22 - a12 * a12;
            T c01 = a01 * b22 - a12 * a02;
            T c02 = a01 * a12 - b11 * a02;
            T det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);

            // The halfDet value is cos(3*theta) mentioned in the PDF. The
            // acos(z) function requires |z| <= 1, but will fail silently
            // and return NaN if the input is larger than 1 in magnitude.
            // To avoid this problem due to rounding errors, the halfDet
            // value is clamped to [-1,1].
            T halfDet = det * (T)0.5;
            halfDet = fminf(fmaxf(halfDet, (T)-1), (T)1);

            // The eigenvalues of B are ordered as
            // beta0 <= beta1 <= beta2.  The number of digits in
            // twoThirdsPi is chosen so that, whether float or double,
            // the floating-point number is the closest to theoretical
            // 2*pi/3.
            T angle = acosf(halfDet) / (T)3;
            T const twoThirdsPi = (T)2.09439510239319549;
            T beta2 = cosf(angle) * (T)2;
            T beta0 = cosf(angle + twoThirdsPi) * (T)2;
            T beta1 = -(beta0 + beta2);

            // The eigenvalues of A are ordered as
            // alpha0 <= alpha1 <= alpha2.
            eval[0] = q + p * beta0;
            eval[1] = q + p * beta1;
            eval[2] = q + p * beta2;

            // Compute the eigenvectors so that the set
            // {evec[0], evec[1], evec[2]} is right handed and
            // orthonormal.
            if (halfDet >= (T)0)
            {
                ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval[2], evec[2]);
                ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[2], eval[1], evec[1]);
                Cross(evec[1], evec[2], evec[0]);
            }
            else
            {
                ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval[0], evec[0]);
                ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[0], eval[1], evec[1]);
                Cross(evec[0], evec[1], evec[2]);
            }
        }
        else
        {
            // The matrix is diagonal.
            eval[0] = a00;
            eval[1] = a11;
            eval[2] = a22;
            evec[0] = { (T)1, (T)0, (T)0 };
            evec[1] = { (T)0, (T)1, (T)0 };
            evec[2] = { (T)0, (T)0, (T)1 };
        }

        // The preconditioning scaled the matrix A, which scales the
        // eigenvalues.  Revert the scaling.
        eval[0] *= maxAbsElement;
        eval[1] *= maxAbsElement;
        eval[2] *= maxAbsElement;

        SortEigenstuff<T>()(sortType, true, eval, evec);
    }

private:
    __device__
    void Multiply(T s, const T (&U)[3], T (&V)[3])
    {
        T product[3] = { s * U[0], s * U[1], s * U[2] };
        copy_vec_elements(V, product);
    }

    __device__
    void Subtract(const T (&U)[3], const T (&V)[3], T (&W)[3])
    {
        T difference[3] = { U[0] - V[0], U[1] - V[1], U[2] - V[2] };
        copy_vec_elements(W, difference);
    }

    __device__
    void Divide(const T (&U)[3], T s, T (&V)[3])
    {
        T invS = (T)1 / s;
        T division[3] = { U[0] * invS, U[1] * invS, U[2] * invS };
        copy_vec_elements(V, division);
    }

    __device__
    T Dot(const T (&U)[3], T s, T (&V)[3])
    {
        T dot = U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
        return dot;
    }

    __device__
    void Cross(const T (&U)[3], const T (&V)[3], T (&W)[3])
    {
        T cross[3] =
                {
                        U[1] * V[2] - U[2] * V[1],
                        U[2] * V[0] - U[0] * V[2],
                        U[0] * V[1] - U[1] * V[0]
                };
        copy_vec_elements(W, cross);

    }

    __device__
    void ComputeOrthogonalComplement(const T (&W)[3],
                                     const T (&U)[3], T (&V)[3])
    {
        // Robustly compute a right-handed orthonormal set { U, V, W }.
        // The vector W is guaranteed to be unit-length, in which case
        // there is no need to worry about a division by zero when
        // computing invLength.
        T invLength;
        if (fabsf(W[0]) > fabsf(W[1]))
        {
            // The component of maximum absolute value is either W[0]
            // or W[2].
            invLength = (T)1 / sqrtf(W[0] * W[0] + W[2] * W[2]);
            U = { -W[2] * invLength, (T)0, +W[0] * invLength };
        }
        else
        {
            // The component of maximum absolute value is either W[1]
            // or W[2].
            invLength = (T)1 / sqrtf(W[1] * W[1] + W[2] * W[2]);
            U = { (T)0, +W[2] * invLength, -W[1] * invLength };
        }
        Cross(W, U, V);
    }

    __device__
    void ComputeEigenvector0(T a00, T a01, T a02, T a11, T a12, T a22,
                             T eval0, T (&evec0)[3])
    {
        // Compute a unit-length eigenvector for eigenvalue[i0].  The
        // matrix is rank 2, so two of the rows are linearly independent.
        // For a robust computation of the eigenvector, select the two
        // rows whose cross product has largest length of all pairs of
        // rows.
        T row0[3] = { a00 - eval0, a01, a02 };
        T row1[3] = { a01, a11 - eval0, a12 };
        T row2[3] = { a02, a12, a22 - eval0 };
        T r0xr1[3]; Cross(row0, row1, r0xr1);
        T r0xr2[3]; Cross(row0, row2, r0xr2);
        T r1xr2[3]; Cross(row1, row2,r1xr2);
        T d0 = Dot(r0xr1, r0xr1);
        T d1 = Dot(r0xr2, r0xr2);
        T d2 = Dot(r1xr2, r1xr2);

        T dmax = d0;
        int imax = 0;
        if (d1 > dmax)
        {
            dmax = d1;
            imax = 1;
        }
        if (d2 > dmax)
        {
            imax = 2;
        }

        if (imax == 0)
        {
             Divide(r0xr1, std::sqrt(d0), evec0);
        }
        else if (imax == 1)
        {
            Divide(r0xr2, std::sqrt(d1), evec0);
        }
        else
        {
            Divide(r1xr2, std::sqrt(d2), evec0);
        }
    }

    __device__
    void ComputeEigenvector1(T a00, T a01, T a02, T a11, T a12, T a22,
                             const T (&evec0)[3], T eval1, T (&evec1)[3])
    {
        // Robustly compute a right-handed orthonormal set
        // { U, V, evec0 }.
        T U[3], V[3];
        ComputeOrthogonalComplement(evec0, U, V);

        // Let e be eval1 and let E be a corresponding eigenvector which
        // is a solution to the linear system (A - e*I)*E = 0.  The matrix
        // (A - e*I) is 3x3, not invertible (so infinitely many
        // solutions), and has rank 2 when eval1 and eval are different.
        // It has rank 1 when eval1 and eval2 are equal.  Numerically, it
        // is difficult to compute robustly the rank of a matrix.  Instead,
        // the 3x3 linear system is reduced to a 2x2 system as follows.
        // Define the 3x2 matrix J = [U V] whose columns are the U and V
        // computed previously.  Define the 2x1 vector X = J*E.  The 2x2
        // system is 0 = M * X = (J^T * (A - e*I) * J) * X where J^T is
        // the transpose of J and M = J^T * (A - e*I) * J is a 2x2 matrix.
        // The system may be written as
        //     +-                        -++-  -+       +-  -+
        //     | U^T*A*U - e  U^T*A*V     || x0 | = e * | x0 |
        //     | V^T*A*U      V^T*A*V - e || x1 |       | x1 |
        //     +-                        -++   -+       +-  -+
        // where X has row entries x0 and x1.

        T AU[3] =
                {
                        a00 * U[0] + a01 * U[1] + a02 * U[2],
                        a01 * U[0] + a11 * U[1] + a12 * U[2],
                        a02 * U[0] + a12 * U[1] + a22 * U[2]
                };

        T AV[3] =
                {
                        a00 * V[0] + a01 * V[1] + a02 * V[2],
                        a01 * V[0] + a11 * V[1] + a12 * V[2],
                        a02 * V[0] + a12 * V[1] + a22 * V[2]
                };

        T m00 = U[0] * AU[0] + U[1] * AU[1] + U[2] * AU[2] - eval1;
        T m01 = U[0] * AV[0] + U[1] * AV[1] + U[2] * AV[2];
        T m11 = V[0] * AV[0] + V[1] * AV[1] + V[2] * AV[2] - eval1;

        // For robustness, choose the largest-length row of M to compute
        // the eigenvector.  The 2-tuple of coefficients of U and V in the
        // assignments to eigenvector[1] lies on a circle, and U and V are
        // unit length and perpendicular, so eigenvector[1] is unit length
        // (within numerical tolerance).
        T absM00 = fabsf(m00);
        T absM01 = fabsf(m01);
        T absM11 = fabsf(m11);
        T maxAbsComp;
        if (absM00 >= absM11)
        {
            maxAbsComp = fmaxf(absM00, absM01);
            if (maxAbsComp > (T)0)
            {
                if (absM00 >= absM01)
                {
                    m01 /= m00;
                    m00 = (T)1 / sqrtf((T)1 + m01 * m01);
                    m01 *= m00;
                }
                else
                {
                    m00 /= m01;
                    m01 = (T)1 / sqrtf((T)1 + m00 * m00);
                    m00 *= m01;
                }
                T t1[3]; Multiply(m01, U, t1);
                T t2[3]; Multiply(m00, V, t2);
                Subtract(t1, t2, evec1);
            }
            else
            {
                copy_vec_elements(evec1, U);
            }
        }
        else
        {
            maxAbsComp = fmaxf(absM11, absM01);
            if (maxAbsComp > (T)0)
            {
                if (absM11 >= absM01)
                {
                    m01 /= m11;
                    m11 = (T)1 / sqrtf((T)1 + m01 * m01);
                    m01 *= m11;
                }
                else
                {
                    m11 /= m01;
                    m01 = (T)1 / sqrtf((T)1 + m11 * m11);
                    m11 *= m01;
                }

                T t1[3]; Multiply(m11, U, t1);
                T t2[3]; Multiply(m01, V, t2);
                evec1 = Subtract(t1, t2, evec1);
            }
            else
            {
                copy_vec_elements(evec1, U);
            }
        }
    }
};

#endif //KNNCUDA_EIGENSOLVER_CUH
