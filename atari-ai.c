#include "atari-ai.h"

/* PDP-11 seed value */
static int g_seed = 887;  

/* Global state definitions */
q16_weights g_q16w;
q8_weights  g_q8w;
gradients   g_grad;
forward_cache g_fwd;
backward_workspace g_bkw;
training_state  g_trn;

/* ===== Lookup tables ===== */

/* Precomputed values in Q8 format for softmax normalization, mapping index i to exp(-i/32) 
where i ranges from 0 to 255. */
const q8 exptbl[256] = {
    /* [0]-[7] */
     256, 248, 240, 233, 226, 219, 212, 206,
    /* [8]-[15] */
     199, 193, 187, 182, 176, 171, 165, 160,
    /* [16]-[23] */
     155, 150, 146, 141, 137, 133, 129, 125,
    /* [24]-[31] */
     121, 117, 114, 110, 107, 103, 100,  97,
    /* [32]-[39] */
      94,  91,  88,  86,  83,  81,  78,  76,
    /* [40]-[47] */
      73,  71,  69,  67,  65,  63,  61,  59,
    /* [48]-[55] */
      57,  55,  54,  52,  50,  49,  47,  46,
    /* [56]-[63] */
      44,  43,  42,  41,  39,  38,  37,  36,
    /* [64]-[71] */
      35,  34,  33,  32,  31,  30,  29,  28,
    /* [72]-[79] */
      27,  26,  25,  25,  24,  23,  22,  22,
    /* [80]-[87] */
      21,  20,  20,  19,  19,  18,  17,  17,
    /* [88]-[95] */
      16,  16,  15,  15,  14,  14,  14,  13,
    /* [96]-[103] */
      13,  12,  12,  12,  11,  11,  11,  10,
    /* [104]-[111] */
      10,  10,   9,   9,   9,   8,   8,   8,
    /* [112]-[119] */
       8,   7,   7,   7,   7,   7,   6,   6,
    /* [120]-[127] */
       6,   6,   6,   5,   5,   5,   5,   5,
    /* [128]-[135] */
       5,   5,   4,   4,   4,   4,   4,   4,
    /* [136]-[143] */
       4,   4,   3,   3,   3,   3,   3,   3,
    /* [144]-[151] */
       3,   3,   3,   3,   3,   2,   2,   2,
    /* [152]-[159] */
       2,   2,   2,   2,   2,   2,   2,   2,
    /* [160]-[167] */
       2,   2,   2,   2,   2,   1,   1,   1,
    /* [168]-[175] */
       1,   1,   1,   1,   1,   1,   1,   1,
    /* [176]-[183] */
       1,   1,   1,   1,   1,   1,   1,   1,
    /* [184]-[191] */
       1,   1,   1,   1,   1,   1,   1,   1,
    /* [192]-[199] */
       1,   1,   1,   1,   1,   1,   1,   1,
    /* [200]-[207] */
       0,   0,   0,   0,   0,   0,   0,   0,
    /* [208]-[215] */
       0,   0,   0,   0,   0,   0,   0,   0,
    /* [216]-[223] */
       0,   0,   0,   0,   0,   0,   0,   0,
    /* [224]-[231] */
       0,   0,   0,   0,   0,   0,   0,   0,
    /* [232]-[239] */
       0,   0,   0,   0,   0,   0,   0,   0,
    /* [240]-[247] */
       0,   0,   0,   0,   0,   0,   0,   0,
    /* [248]-[255] */
       0,   0,   0,   0,   0,   0,   0,   0,
};

/* Precomputed values in Q12 format for logarithm, mapping index i to log(i/256) 
where i ranges from 0 to 256. */
const q12 logtbl[257] = {
    22713, 22713, 19874, 18213, 17035, 16121, 15374, 14743,
    14196, 13713, 13282, 12891, 12535, 12207, 11903, 11621,
    11357, 11108, 10874, 10653, 10443, 10243, 10052,  9870,
     9696,  9529,  9368,  9213,  9064,  8921,  8782,  8647,
     8517,  8391,  8269,  8150,  8035,  7923,  7813,  7707,
     7603,  7502,  7404,  7307,  7213,  7121,  7031,  6943,
     6857,  6772,  6689,  6608,  6529,  6451,  6374,  6299,
     6225,  6153,  6081,  6011,  5943,  5875,  5808,  5743,
     5678,  5615,  5552,  5491,  5430,  5370,  5311,  5253,
     5196,  5139,  5084,  5029,  4974,  4921,  4868,  4816,
     4764,  4713,  4663,  4613,  4564,  4516,  4468,  4421,
     4374,  4328,  4282,  4237,  4192,  4148,  4104,  4060,
     4017,  3975,  3933,  3891,  3850,  3810,  3769,  3729,
     3690,  3650,  3612,  3573,  3535,  3497,  3460,  3423,
     3386,  3350,  3314,  3278,  3242,  3207,  3172,  3138,
     3103,  3069,  3036,  3002,  2969,  2936,  2904,  2871,
     2839,  2807,  2776,  2744,  2713,  2682,  2651,  2621,
     2591,  2561,  2531,  2501,  2472,  2443,  2414,  2385,
     2357,  2328,  2300,  2272,  2244,  2217,  2189,  2162,
     2135,  2108,  2082,  2055,  2029,  2003,  1977,  1951,
     1925,  1900,  1874,  1849,  1824,  1799,  1774,  1750,
     1725,  1701,  1677,  1653,  1629,  1605,  1582,  1558,
     1535,  1512,  1488,  1466,  1443,  1420,  1397,  1375,
     1353,  1330,  1308,  1286,  1265,  1243,  1221,  1200,
     1178,  1157,  1136,  1115,  1094,  1073,  1052,  1032,
     1011,   991,   970,   950,   930,   910,   890,   870,
      850,   831,   811,   792,   772,   753,   734,   715,
      696,   677,   658,   639,   621,   602,   584,   565,
      547,   529,   511,   492,   474,   457,   439,   421,
      403,   386,   368,   351,   333,   316,   299,   281,
      264,   247,   230,   213,   197,   180,   163,   147,
      130,   114,    97,    81,    65,    48,    32,    16,
        0,
};

/* ===== FXMATH.MAC — Level 0 scalar primitives ===== */

/*
fxdiv — Q8 fixed-point division: computes dividend / divisor and returns the quotient in Q8 format.

Inputs:
  dividend  — Q8 fixed-point value (8 integer bits, 8 fractional bits)
  divisor   — Q8 fixed-point value (must not be zero)

Return:
  Q8 fixed-point quotient = dividend / divisor

Method:
  The dividend is shifted left by 8 bits to promote it to Q16, then divided
  by the divisor. The result is truncated to Q8. This avoids the need for
  a hardware floating-point unit on the Atari 8-bit (6502), using only
  integer shift and division operations.

Special case:
  If divisor is zero, returns 0 to avoid division-by-zero.
*/

q8 fxdiv(q8 dividend, q8 divisor)
{
    int32_t num;
    if (divisor == 0) return 0;
    num = (int32_t)dividend << 8;
    return (q8)(num / (int32_t)divisor);
}

/*
fxmul — Q8 fixed-point multiplication: computes the product of two Q8 values and returns the result in Q16 format.

Inputs:
  a — Q8 fixed-point value (8 integer bits, 8 fractional bits)
  b — Q8 fixed-point value (8 integer bits, 8 fractional bits)

Return:
  Q16 fixed-point product = a * b

Method:
  Multiplies the two Q8 values using 32-bit arithmetic to produce a 32-bit
  product (Q24 format), then shifts right by 8 bits to obtain the result
  in Q16 format. This emulates the PDP-11 MUL instruction (which produces
  a 32-bit result in a register pair) followed by ASHC $-8 to shift the
  high word right by 8 bits.

  Q8 * Q8 -> 32-bit product (Q24) -> Q16 result
*/
q16 fxmul(q8 a, q8 b)
{
    /* PDP-11: mul b, a  -> R0:R1 = a*b (Q24)
                ashc $-8, r0  -> R0 = Q8 result (high word) */
    int32_t product = (int32_t)a * (int32_t)b;  /* Q24 */
    return (q16)(product >> 8);                   /* Q8 in high word */
}

/* ===== VECOP.MAC — Level 1 vector operations ===== */

/*
vdot — Vector dot product: computes the sum of element-wise products of two Q8 vectors.

Inputs:
  x     — pointer to first Q8 vector
  y     — pointer to second Q8 vector (same length as x)
  len   — number of elements in each vector

Return:
  Q8 fixed-point dot product = sum(x[i] * y[i]) for i = 0 to len-1

Method:
  Multiplies corresponding elements of the two vectors, accumulating the
  32-bit products using a low/high split (dtlo/dthi) to avoid overflow.
  The final 32-bit accumulator is shifted right by 8 bits to produce a
  Q8 result. The result is clamped to [-32768, 32767] to prevent overflow.

  Q8 * Q8 -> 32-bit product -> accumulate -> Q8 result (clamped)
*/
q8 vdot(const q8 *x, const q8 *y, int len)
{
    uint16_t dtlo = 0;
    uint16_t dthi = 0;
    int32_t acc;
    int32_t result;
    int32_t prod;
    uint32_t lo_sum;
    uint16_t carry;
    int i;

    for (i = 0; i < len; i++) {
        prod = (int32_t)x[i] * (int32_t)y[i];
        lo_sum = (uint32_t)dtlo + (uint32_t)(uint16_t)prod;
        carry = (uint16_t)(lo_sum >> 16);
        dtlo = (uint16_t)lo_sum;
        dthi = (uint16_t)(dthi + carry + (uint16_t)(int16_t)(prod >> 16));
    }

    acc = ((int32_t)(int16_t)dthi << 16) | (uint32_t)dtlo;
    result = acc >> 8;
    if (result > 32767L)  result = 32767L;
    if (result < -32768L) result = -32768L;
    return (q8)result;
}

/*
vadd — Vector addition: computes the element-wise sum of two Q8 vectors.

Inputs:
  x     — pointer to first Q8 vector
  y     — pointer to second Q8 vector (same length as x)
  z     — pointer to output Q8 vector (same length as x and y)
  len   — number of elements in each vector

Return:
  None (result stored in-place in z)

Method:
  Adds corresponding elements of x and y, storing the result in z.
  Uses 32-bit arithmetic to prevent overflow: two Q8 values can sum to
  +/-65534, which overflows a 16-bit int. Each element is computed as
  z[i] = x[i] + y[i] for i = 0 to len-1.

  Q8 + Q8 -> 32-bit sum -> Q8 result (stored in z)
*/
void vadd(const q8 *x, const q8 *y, q8 *z, int len)
{
    /* PDP-11: z[i] = x[i]; z[i] += y[i].
       Use int32_t: two Q8 values can sum to ±65534, overflowing 16-bit int. */
    int i = 0;
    for (i = 0; i < len; i++)
        z[i] = (q8)((int32_t)x[i] + (int32_t)y[i]);
}

/*
vsub — Vector subtraction: computes the element-wise difference of two Q8 vectors.

Inputs:
  x     — pointer to first Q8 vector (minuend)
  y     — pointer to second Q8 vector (subtrahend, same length as x)
  z     — pointer to output Q8 vector (same length as x and y)
  len   — number of elements in each vector

Return:
  None (result stored in-place in z)

Method:
  Subtracts corresponding elements of y from x, storing the result in z.
  Uses 32-bit arithmetic to prevent overflow: the difference of two Q8
  values can span +/-65535, which overflows a 16-bit int. Each element
  is computed as z[i] = x[i] - y[i] for i = 0 to len-1.

  Q8 - Q8 -> 32-bit difference -> Q8 result (stored in z)
*/
void vsub(const q8 *x, const q8 *y, q8 *z, int len)
{
    /* PDP-11: z[i] = x[i]; z[i] -= y[i].
       Use int32_t: difference of two Q8 values spans ±65535, overflowing 16-bit int. */
    int i = 0;
    for (i = 0; i < len; i++)
        z[i] = (q8)((int32_t)x[i] - (int32_t)y[i]);
}

/*
vscl — Vector scaling: multiplies each element of a Q8 vector by a Q8 scalar.

Inputs:
  x     — pointer to input Q8 vector
  y     — pointer to output Q8 vector (same length as x)
  len   — number of elements in the vector
  alpha — Q8 scalar to multiply each element by

Return:
  None (result stored in-place in y)

Method:
  Multiplies each element of x by alpha, shifts the 32-bit product right by 8 bits
  to produce a Q8 result, and stores it in y.
*/
void vscl(const q8 *x, q8 *y, int len, q8 alpha)
{
    /* PDP-11: y[i] = (alpha * x[i]) >> 8 */
    int i = 0;
    for (i = 0; i < len; i++) {
        int32_t prod = (int32_t)alpha * (int32_t)x[i];
        y[i] = (q8)(prod >> 8);
    }
}

/*
vmax — Vector maximum: finds the maximum value in a Q8 vector and its index.

Inputs:
  vec     — pointer to input Q8 vector
  len     — number of elements in the vector
  out_idx — pointer to store index of maximum value (optional, can be NULL)

Return:
  Maximum value in the vector (Q8)

Method:
  Iterates through the vector, keeping track of the maximum value and its index.
*/
int vmax(const q8 *vec, int len, int *out_idx)
{
    /* PDP-11: R0=max value, R1=index of max */
    int max_val = (int16_t)vec[0];
    int max_idx = 0;

    int i = 1;
    for (i = 1; i < len; i++) {
        int val = (int16_t)vec[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    if (out_idx) *out_idx = max_idx;
    return max_val;
}

/*
vcpy — Vector copy: copies elements from one Q8 vector to another.

Inputs:
  src   — pointer to source Q8 vector
  dst   — pointer to destination Q8 vector (same length as src)
  len   — number of elements in the vector

Return:
  None (result stored in-place in dst)

Method:
  Copies each element from src to dst.
*/
void vcpy(const q8 *src, q8 *dst, int len)
{
    int i = 0;
    for (i = 0; i < len; i++)
        dst[i] = src[i];
}

/*
vclr — Vector clear: sets all elements of a Q8 vector to zero.

Inputs:
  vec   — pointer to Q8 vector
  len   — number of elements in the vector

Return:
  None (result stored in-place in vec)

Method:
  Sets each element of vec to zero.
*/
void vclr(q8 *vec, int len)
{
    int i = 0;
    for (i = 0; i < len; i++)
        vec[i] = 0;
}

/*
vsadd — Vector scaled accumulate: multiplies each element of src by a Q8 scalar and adds the result to the corresponding element of dst.
 
Inputs:
  src    — pointer to input Q8 vector
  dst    — pointer to output Q8 vector (same length as src)
  len    — number of elements in the vector
  scalar — Q8 scalar to add to each element

Return:
  None (result stored in-place in dst)

Method:
  Multiplies each element of src by scalar, shifts the 32-bit product right by 8 bits
  to produce a Q8 result, and adds it to the corresponding element in dst with saturation.
*/
void vsadd(const q8 *src, q8 *dst, int len, q8 scalar)
{
    int i = 0;
    for (i = 0; i < len; i++) {
        int32_t prod = (int32_t)scalar * (int32_t)src[i];
        int32_t val = prod >> 8;
        int32_t sum;                   /* hoisted before the if-statements */

        if (val > 32767L) val = 32767L;
        if (val < -32768L) val = -32768L;

        sum = (int16_t)dst[i] + val;   /* assignment, not declaration */
        if (sum > 32767L) sum = 32767L;
        if (sum < -32768L) sum = -32768L;
        dst[i] = (q8)sum;
    }
}

/* ===== MATOP.MAC — Level 2 matrix-vector operations ===== */

/*
mvmul — Matrix-vector multiplication: computes the product of a Q8 matrix and a Q8 vector.

Inputs:
  mat    — pointer to Q8 matrix stored in row-major order (rows x cols)
  vin    — pointer to input Q8 vector (length = cols)
  vout   — pointer to output Q8 vector (length = rows)
  rows   — number of rows in the matrix
  cols   — number of columns in the matrix

Return:
  None (result stored in-place in vout)

Method:
  For each row i of the matrix, computes the dot product of row i with vin,
  accumulating 32-bit products using a low/high split (dtlo/dthi) to avoid
  overflow. The final 32-bit accumulator is shifted right by 8 bits to produce
  a Q8 result, which is clamped to [-32768, 32767]. Each element is computed
  as vout[i] = sum(mat[i*cols+j] * vin[j]) for j = 0 to cols-1.

  Q8 matrix * Q8 vector -> 32-bit accumulation -> Q8 result (stored in vout)
*/
void mvmul(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols)
{
    int i = 0;
    for (i = 0; i < rows; i++) {
        uint16_t dtlo = 0;
        uint16_t dthi = 0;
        int32_t acc;
        int32_t result;
        int j = 0;
        for (j = 0; j < cols; j++) {
            int32_t prod = (int32_t)mat[i * cols + j] * (int32_t)vin[j];
            uint32_t lo_sum = (uint32_t)dtlo + (uint32_t)(uint16_t)prod;
            uint16_t carry = (uint16_t)(lo_sum >> 16);
            dtlo = (uint16_t)lo_sum;
            dthi = (uint16_t)(dthi + carry + (uint16_t)(int16_t)(prod >> 16));
        }
        acc = ((int32_t)(int16_t)dthi << 16) | (uint32_t)dtlo;
        result = acc >> 8;
        if (result > 32767L)  result = 32767L;
        if (result < -32768L) result = -32768L;
        vout[i] = (q8)result;
    }
}

/*
mvadd — Matrix-vector multiply-add: computes vout[i] = vout[i] + sum(mat[i*cols+j] * vin[j]) for each row.

Inputs:
  mat    — pointer to Q8 matrix stored in row-major order (rows x cols)
  vin    — pointer to input Q8 vector (length = cols)
  vout   — pointer to output Q8 vector (length = rows), modified in-place
  rows   — number of rows in the matrix
  cols   — number of columns in the matrix

Return:
  None (result stored in-place in vout)

Method:
  For each row i of the matrix, computes the dot product of row i with vin,
  accumulating 32-bit products using a low/high split (dtlo/dthi) to avoid
  overflow. The final 32-bit accumulator is shifted right by 8 bits to produce
  a Q8 result, which is clamped to [-32768, 32767]. This result is then added
  to the existing value in vout[i] with saturation clamping. Each element is
  computed as vout[i] = vout[i] + sum(mat[i*cols+j] * vin[j]) for j = 0 to
  cols-1.

  Q8 matrix * Q8 vector + Q8 vector -> 32-bit accumulation -> Q8 result (stored in vout)
*/
void mvadd(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols)
{
    int i = 0;
    for (i = 0; i < rows; i++) {
        uint16_t dtlo = 0;
        uint16_t dthi = 0;
        int32_t acc;
        int32_t result;
        int32_t sum;
        int j = 0;
        for (j = 0; j < cols; j++) {
            int32_t prod = (int32_t)mat[i * cols + j] * (int32_t)vin[j];
            uint32_t lo_sum = (uint32_t)dtlo + (uint32_t)(uint16_t)prod;
            uint16_t carry = (uint16_t)(lo_sum >> 16);
            dtlo = (uint16_t)lo_sum;
            dthi = (uint16_t)(dthi + carry + (uint16_t)(int16_t)(prod >> 16));
        }
        acc = ((int32_t)(int16_t)dthi << 16) | (uint32_t)dtlo;
        result = acc >> 8;
        if (result > 32767L)  result = 32767L;
        if (result < -32768L) result = -32768L;
        sum = (int16_t)vout[i] + result;
        if (sum > 32767L)  sum = 32767L;
        if (sum < -32768L) sum = -32768L;
        vout[i] = (q8)sum;
    }
}

/*
vtmul — Transposed matrix-vector multiplication: computes vout = mat^T * vin using accumulated saturation arithmetic.

Inputs:
  mat    — pointer to Q8 matrix stored in row-major order (rows x cols)
  vin    — pointer to input Q8 vector (length = rows)
  vout   — pointer to output Q8 vector (length = cols), cleared to zero before computation
  rows   — number of rows in the matrix
  cols   — number of columns in the matrix

Return:
  None (result stored in-place in vout)

Method:
  First clears vout to zero. Then for each row i of the matrix, uses vin[i]
  as a scalar to scale the entire row, accumulating the scaled values into vout
  with saturation clamping at each step. Each output element is computed as
  vout[j] = sum(vin[i] * mat[i*cols+j]) for i = 0 to rows-1, with intermediate
  values clamped to [-32768, 32767] to prevent overflow.

  Q8 matrix^T * Q8 vector -> 32-bit accumulation with saturation -> Q8 result (stored in vout)
*/
void vtmul(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols)
{
    int i = 0;
    int j = 0;
    for (j = 0; j < cols; j++)
        vout[j] = 0;

    for (i = 0; i < rows; i++) {
        q8 scalar = vin[i];
        int32_t sum;
        for (j = 0; j < cols; j++) {
            int32_t prod = (int32_t)mat[i * cols + j] * (int32_t)scalar;
            int32_t val = prod >> 8;
            if (val > 32767L) val = 32767L;
            if (val < -32768L) val = -32768L;
            sum = (int16_t)vout[j] + val;
            if (sum > 32767L) sum = 32767L;
            if (sum < -32768L) sum = -32768L;
            vout[j] = (q8)sum;
        }
    }
}
/*
outer — Outer product add: computes the outer product of two Q8 vectors and adds it to a Q8 matrix.

Inputs:
  vx    — pointer to first Q8 vector (length = rows)
  vy    — pointer to second Q8 vector (length = cols)
  mat   — pointer to Q8 matrix stored in row-major order (rows x cols), modified in-place
  rows  — number of rows (length of vx)
  cols  — number of columns (length of vy)

Return:
  None (result stored in-place in mat)

Method:
  For each row i of the matrix, uses vx[i] as a scalar to scale the entire
  vector vy, then adds the scaled result to the corresponding row of mat with
  saturation clamping at each step. Each element is computed as
  mat[i*cols+j] = mat[i*cols+j] + (vx[i] * vy[j] >> 8) for i = 0 to rows-1
  and j = 0 to cols-1, with intermediate values clamped to [-32768, 32767]
  to prevent overflow.

  Q8 vector outer Q8 vector + Q8 matrix -> 32-bit accumulation with saturation -> Q8 result (stored in mat)
*/
void outer(const q8 *vx, const q8 *vy, q8 *mat, int rows, int cols)
{
    int i = 0;
    for (i = 0; i < rows; i++) {
        q8 scalar = vx[i];
        int j = 0;
        for (j = 0; j < cols; j++) {
            int32_t prod = (int32_t)scalar * (int32_t)vy[j];
            int32_t val = prod >> 8;
            int32_t sum;
            if (val > 32767L) val = 32767L;
            if (val < -32768L) val = -32768L;
            sum = (int16_t)mat[i * cols + j] + val;
            if (sum > 32767L) sum = 32767L;
            if (sum < -32768L) sum = -32768L;
            mat[i * cols + j] = (q8)sum;
        }
    }
}

/* ===== ACTFN.MAC — Activation functions ===== */

/*
vrelu — Vector ReLU activation: applies the Rectified Linear Unit function element-wise to a Q8 vector.

Inputs:
  vec   — pointer to Q8 vector to modify in-place
  len   — number of elements in the vector

Return:
  None (result stored in-place in vec)

Method:
  Iterates through the vector and sets each negative element to zero.
  Positive elements are left unchanged. This implements the standard
  ReLU activation function: f(x) = max(0, x).
*/
void vrelu(q8 *vec, int len)
{
    /* PDP-11: negative elements zeroed, positive kept */
    int i = 0;
    for (i = 0; i < len; i++)
        if ((int16_t)vec[i] < 0)
            vec[i] = 0;
}

/*
sftmx — Softmax activation: applies the softmax function to a Q8 vector.

Inputs:
  vec   — pointer to Q8 vector to modify in-place
  len   — number of elements in the vector

Return:
  None (result stored in-place in vec)

Method:
  Computes the softmax function in a numerically stable way by subtracting
  the maximum value before exponentiation. Uses a lookup table for the
  exponential function and normalizes the result by dividing by the sum.
*/
void sftmx(q8 *vec, int len)
{
    int32_t max_val = (int32_t)(int16_t)vec[0];
    int32_t sum = 0;
    int i = 1;
    for (i = 1; i < len; i++) {
        int32_t val = (int32_t)(int16_t)vec[i];
        if (val > max_val)
            max_val = val;
    }

    /* Step 2: exp(x_i - max) via LUT, accumulate sum */
    for (i = 0; i < len; i++) {
        int32_t diff = (int32_t)(int16_t)vec[i] - max_val;
        int idx = (int)((-diff) >> 3);
        int exp_val;
        if (idx > 255) idx = 255;
        exp_val = exptbl[idx];
        vec[i] = (q8)exp_val;
        sum += exp_val;
    }

    /* Step 3: Normalize — divide each exp value by sum */
    for (i = 0; i < len; i++)
        vec[i] = fxdiv(vec[i], (q8)sum);
}

/* ===== LAYER.MAC — Neural network layers ===== */

/*
embed — Token and position embedding: computes the sum of token embeddings and positional embeddings.

Inputs:
  tokens  — pointer to array of token IDs (integers)
  tkemb   — pointer to token embedding matrix in row-major order (vocab x d_model)
  psemb   — pointer to positional embedding matrix in row-major order (seq_len x d_model)
  xout    — pointer to output Q8 vector (same length as tokens, modified in-place)
  seq_len — number of tokens in the sequence
  d_model — dimension of the embedding vectors

Return:
  None (result stored in-place in xout)

Method:
  For each position i in the sequence, looks up the token embedding for
  tokens[i] from tkemb and adds the positional embedding from psemb. Each
  output element is computed as xout[i*d_model+j] = tkemb[tokens[i]*d_model+j]
  + psemb[i*d_model+j] for i = 0 to seq_len-1 and j = 0 to d_model-1.
*/
void embed(const int *tokens, const q8 *tkemb, const q8 *psemb,
           q8 *xout, int seq_len, int d_model)
{
    /* PDP-11: output[i] = token_embed[tokens[i]] + pos_embed[i]
                Row size = d_model * 2 bytes */
    int i = 0;
    for (i = 0; i < seq_len; i++) {
        int tok_id = tokens[i];
        const q8 *tok_row = tkemb + tok_id * d_model;
        q8 *out_row = xout + i * d_model;

        int j = 0;
        for (j = 0; j < d_model; j++)
            out_row[j] = (q8)((int32_t)tok_row[j] + (int32_t)psemb[i * d_model + j]);
    }
}

/*
proj — Projection layer: computes logits by multiplying input with output weights.

Inputs:
  yin    — pointer to input Q8 vector (sequence of embeddings)
  wout   — pointer to output weight matrix in row-major order (d_model x vocab)
  logits — pointer to output Q8 vector (seq_len x vocab, modified in-place)
  seq_len — number of tokens in the sequence
  d_model — dimension of the embedding vectors
  vocab   — size of the vocabulary

Return:
  None (result stored in-place in logits)

Method:
  For each position i in the sequence, computes logits[i] = Wout^T * Y[i].
*/
void proj(const q8 *yin, const q8 *wout, q8 *logits,
          int seq_len, int d_model, int vocab)
{
    /* PDP-11: logits[i] = Wout^T * Y[i] for each position.
                Uses self-modifying code to patch VTMUL params. */
    int i = 0;
    for (i = 0; i < seq_len; i++) {
        vtmul(wout, yin, logits, d_model, vocab);
        yin += d_model;
        logits += vocab;
    }
}

/*
attn — Self-attention forward pass: computes scaled dot-product attention with residual connection.

Inputs:
  xin    — pointer to input Q8 vector (seq_len x d_model)
  wq     — pointer to query weight matrix in row-major order (d_model x d_model)
  wk     — pointer to key weight matrix in row-major order (d_model x d_model)
  wv     — pointer to value weight matrix in row-major order (d_model x d_model)
  yout   — pointer to output Q8 vector (seq_len x d_model, modified in-place)
  work   — workspace buffer with layout Q | K | V | S (Q, K, V are seq_len x d_model; S is seq_len x seq_len)
  seq_len — number of tokens in the sequence
  d_model — dimension of the embedding vectors
  sqrtsh — shift value used to divide attention scores by sqrt(d_model)

Return:
  None (result stored in-place in yout)

Method:
  Performs a 7-step self-attention forward pass:
  1. Project input to Q, K, V: Q = X.Wq, K = X.Wk, V = X.Wv
  2. Compute attention scores: S[i][j] = dot(Q[i], K[j]) / sqrt(d)
  3. Apply softmax to each row of S
  4. Compute attended values: O[i] = V^T . S[i]
  5. Add residual connection: Y[i] = O[i] + X[i]
*/
void attn(const q8 *xin, const q8 *wq, const q8 *wk, const q8 *wv,
          q8 *yout, q8 *work, int seq_len, int d_model, int sqrtsh)
{
    /* PDP-11: 7-step self-attention forward pass.
                1. Q=X.Wq, K=X.Wk, V=X.Wv
                2. S[i][j] = Q[i].K[j] / sqrt(d)
                3. A[i] = softmax(S[i])
                4. O[i] = V^T . S[i]
                5. Y[i] = O[i] + X[i] (residual) */

    /* Workspace layout: Q | K | V | S */
    q8 *Q = work;
    q8 *K = Q + seq_len * d_model;
    q8 *V = K + seq_len * d_model;
    q8 *S = V + seq_len * d_model;

    /* Negate shift for ASR (PDP-11 uses neg at_shf before ASH) */
    int shift = -sqrtsh;

    // Index variables for loops
    int i = 0;
    int j = 0;
    /* Steps 1-3: Q=X.Wq, K=X.Wk, V=X.Wv — per-position vtmul loop
       PDP-11 at_bpr: loops seq_len times calling vtmul(W, X[s], out[s], d, d) */
    int s = 0;
    for (s = 0; s < seq_len; s++) {
        vtmul(wq, xin + s * d_model, Q + s * d_model, d_model, d_model);
        vtmul(wk, xin + s * d_model, K + s * d_model, d_model, d_model);
        vtmul(wv, xin + s * d_model, V + s * d_model, d_model, d_model);
    }

    /* Step 4: S[i][j] = Q[i].K[j] / sqrt(d) */
    for (i = 0; i < seq_len; i++) {
        for (j = 0; j < seq_len; j++) {
            q8 dot = vdot(Q + i * d_model, K + j * d_model, d_model);
            dot = (q8)((int16_t)dot >> (-shift));  /* divide by sqrt(d) */
            S[i * seq_len + j] = dot;
        }
    }

    /* Step 5: softmax per row */
    for (i = 0; i < seq_len; i++)
        sftmx(S + i * seq_len, seq_len);

    /* Step 6: Y[i] = V^T . S[i] */
    for (i = 0; i < seq_len; i++)
        vtmul(V, S + i * seq_len, yout + i * d_model, seq_len, d_model);

    /* Step 7: Y += X (residual).
       Use int32_t: Q8 + Q8 can reach ±65534, overflowing 16-bit int. */
    for (i = 0; i < seq_len * d_model; i++)
        yout[i] = (q8)((int32_t)yout[i] + (int32_t)xin[i]);
}

/* ===== FORWRD.MAC — Forward pass & weight conversion ===== */

/*
cvt16 — Weight conversion: converts Q16 weights (stored as hi/lo pairs) to Q8 format.

Inputs:
  None

Return:
  None

Method:
  For each weight group (tok_emb, pos_emb, wq, wk, wv, wot), reconstructs
  the 32-bit Q16 value from its hi/lo pair, shifts right by 8 bits to obtain
  Q8 format, and stores the result in g_q8w. This prepares the weights for
  use in the forward and backward passes.
*/
void cvt16(void)
{
    /* PDP-11: Convert Q16 weights (hi/lo pairs) to Q8 for forward/backward.
                For each weight: Q16 >> 8 = Q8. */

    /* tok_emb */
    int i = 0;
    for (i = 0; i < TOK_EMB_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.tok_emb.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.tok_emb.lo[i]);
        g_q8w.tok_emb[i] = (q8)(q16 >> 8);
    }
    /* pos_emb */
    for (i = 0; i < POS_EMB_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.pos_emb.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.pos_emb.lo[i]);
        g_q8w.pos_emb[i] = (q8)(q16 >> 8);
    }
    /* Wq */
    for (i = 0; i < WQ_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wq.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wq.lo[i]);
        g_q8w.wq[i] = (q8)(q16 >> 8);
    }
    /* Wk */
    for (i = 0; i < WK_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wk.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wk.lo[i]);
        g_q8w.wk[i] = (q8)(q16 >> 8);
    }
    /* Wv */
    for (i = 0; i < WV_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wv.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wv.lo[i]);
        g_q8w.wv[i] = (q8)(q16 >> 8);
    }
    /* Wout */
    for (i = 0; i < WOT_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wot.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wot.lo[i]);
        g_q8w.wot[i] = (q8)(q16 >> 8);
    }
}

/*
forwrd — Forward pass: executes the complete neural network forward pass.

Inputs:
  None

Return:
  None

Method:
  Executes three stages in sequence:
  1. Embedding: computes token + positional embeddings via embed()
  2. Self-attention: computes attention with residual connection via attn()
  3. Output projection: computes logits via proj()
  Results are stored in the forward cache (g_fwd).
*/
void forwrd(void)
{
    /* PDP-11: 1. Embedding, 2. Self-attention, 3. Output projection */
    embed(g_trn.tokens, g_q8w.tok_emb, g_q8w.pos_emb, g_fwd.xx, SEQ_LEN, D_MODEL);
    attn(g_fwd.xx, g_q8w.wq, g_q8w.wk, g_q8w.wv, g_fwd.yy, g_fwd.work,
         SEQ_LEN, D_MODEL, 2);  /* sqrtsh=2 for d_model=16 */
    proj(g_fwd.yy, g_q8w.wot, g_fwd.logits, SEQ_LEN, D_MODEL, VOCAB);
}

/* ===== BKWRD.MAC — Backward pass ===== */

/*
bkwrd — Backward pass: computes gradients for all network weights via backpropagation.

Inputs:
  None

Return:
  None

Method:
  Performs a 6-step backward pass through the network:
  1. Compute dLogits, dWout, dY: softmax gradient, outer product for Wout
     gradient, and matrix multiply for output gradient
  2. Backward through O = V^T . A: compute dA and dV gradients
  3. Backward through softmax: compute dSc (attention score gradients)
  4. Backward through Q.K^T: compute dQ and dK gradients
  5. Backward through projections: accumulate gradients into dX via mvadd
     and compute weight gradients via outer
  6. Backward through embedding: accumulate gradients into token and
     positional embedding gradients
  Results are stored in the gradient arrays (g_grad).
*/
void bkwrd(void)
{
    q8 *Q = g_fwd.work;
    q8 *K = Q + SEQ_LEN * D_MODEL;
    q8 *V = K + SEQ_LEN * D_MODEL;
    q8 *A = V + SEQ_LEN * D_MODEL;
    int i = 0;
    int j = 0;
    int k = 0;
    int tok;
    int32_t v;
    q15 dot;
    q8 a_ij;
    q15 dot_ad;
    q15 da_val;
    int32_t diff;
    int32_t prod;
    int32_t val;
    int32_t sum;

    /* === Step 1: dLogits, dWout, dY === */
    vclr(g_bkw.dy, SEQ_LEN * D_MODEL);

    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < VOCAB; j++)
            g_bkw.dl[j] = g_fwd.logits[i * VOCAB + j];

        sftmx(g_bkw.dl, VOCAB);

        g_bkw.dl[g_trn.target[i]] = (q8)((int32_t)(int16_t)g_bkw.dl[g_trn.target[i]] - 256);

        for (j = 0; j < VOCAB; j++) {
            v = (int32_t)(int16_t)g_bkw.dl[j] << 7;
            if (v > 32767L)  v = 32767L;
            if (v < -32768L) v = -32768L;
            g_bkw.dl[j] = (q15)v;
        }

        outer(g_fwd.yy + i * D_MODEL, g_bkw.dl, g_grad.dwot, D_MODEL, VOCAB);
        mvmul(g_q8w.wot, g_bkw.dl, g_bkw.dy + i * D_MODEL, D_MODEL, VOCAB);
    }

    /* === Step 2: Backward O = A.V -> dA, dV === */
    vclr(g_bkw.dvv, SEQ_LEN * D_MODEL);

    for (i = 0; i < SEQ_LEN; i++) {
        for (j = 0; j < SEQ_LEN; j++) {
            dot = (q15)vdot(V + j * D_MODEL, g_bkw.dy + i * D_MODEL, D_MODEL);
            g_bkw.da[i * SEQ_LEN + j] = dot;

            a_ij = A[i * SEQ_LEN + j];
            vsadd(g_bkw.dy + i * D_MODEL, g_bkw.dvv + j * D_MODEL, D_MODEL, a_ij);
        }
    }

    /* === Step 3: Backward softmax -> dSc (stored in DA) === */
    for (i = 0; i < SEQ_LEN; i++) {
        dot_ad = (q15)vdot(A + i * SEQ_LEN, g_bkw.da + i * SEQ_LEN, SEQ_LEN);

        for (j = 0; j < SEQ_LEN; j++) {
            da_val = g_bkw.da[i * SEQ_LEN + j];
            diff = (int32_t)(int16_t)da_val - (int32_t)(int16_t)dot_ad;

            if (diff > 32767L)  diff = 32767L;
            if (diff < -32768L) diff = -32768L;

            prod = (int32_t)A[i * SEQ_LEN + j] * diff;
            val = prod >> 8;
            val >>= 2;

            g_bkw.da[i * SEQ_LEN + j] = (q15)val;
        }
    }

    /* === Step 4: Backward Q.K^T -> dQ, dK === */
    for (i = 0; i < SEQ_LEN; i++)
        vtmul(K, g_bkw.da + i * SEQ_LEN, g_bkw.dqq + i * D_MODEL, SEQ_LEN, D_MODEL);

    for (j = 0; j < SEQ_LEN; j++) {
        for (i = 0; i < SEQ_LEN; i++)
            g_bkw.dtmp[i] = g_bkw.da[i * SEQ_LEN + j];

        vtmul(Q, g_bkw.dtmp, g_bkw.dkk + j * D_MODEL, SEQ_LEN, D_MODEL);
    }

    /* === Step 5: Backward projections + dX === */
    vcpy(g_bkw.dy, g_bkw.dxx, SEQ_LEN * D_MODEL);

    for (i = 0; i < SEQ_LEN; i++) {
        mvadd(g_q8w.wq, g_bkw.dqq + i * D_MODEL, g_bkw.dxx + i * D_MODEL, D_MODEL, D_MODEL);
        outer(g_fwd.xx + i * D_MODEL, g_bkw.dqq + i * D_MODEL, g_grad.dwq, D_MODEL, D_MODEL);

        mvadd(g_q8w.wk, g_bkw.dkk + i * D_MODEL, g_bkw.dxx + i * D_MODEL, D_MODEL, D_MODEL);
        outer(g_fwd.xx + i * D_MODEL, g_bkw.dkk + i * D_MODEL, g_grad.dwk, D_MODEL, D_MODEL);

        mvadd(g_q8w.wv, g_bkw.dvv + i * D_MODEL, g_bkw.dxx + i * D_MODEL, D_MODEL, D_MODEL);
        outer(g_fwd.xx + i * D_MODEL, g_bkw.dvv + i * D_MODEL, g_grad.dwv, D_MODEL, D_MODEL);
    }

    /* === Step 6: Backward embedding === */
    for (i = 0; i < SEQ_LEN; i++) {
        tok = g_trn.tokens[i];
        for (k = 0; k < D_MODEL; k++) {
            sum = (int32_t)(int16_t)g_grad.dtke[tok * D_MODEL + k]
                + (int32_t)(int16_t)g_bkw.dxx[i * D_MODEL + k];
            if (sum > 32767L)  sum = 32767L;
            if (sum < -32768L) sum = -32768L;
            g_grad.dtke[tok * D_MODEL + k] = (q15)sum;
        }

        for (k = 0; k < D_MODEL; k++) {
            sum = (int32_t)(int16_t)g_grad.dpse[i * D_MODEL + k]
                + (int32_t)(int16_t)g_bkw.dxx[i * D_MODEL + k];
            if (sum > 32767L)  sum = 32767L;
            if (sum < -32768L) sum = -32768L;
            g_grad.dpse[i * D_MODEL + k] = (q15)sum;
        }
    }
}

/* ===== UPDAT.MAC — Weight init, SGD update, gradient zeroing ===== */
/*
rand_seed — Sets the seed for the pseudo-random number generator.

Inputs:
  seed  — initial seed value for the LCG random number generator

Return:
  None

Method:
  Stores the provided seed value in the global variable g_seed. This
  initializes the linear congruential generator (LCG) used by rand_next()
  to produce reproducible sequences of pseudo-random numbers.
*/
void rand_seed(int seed)
{
    g_seed = seed;
}

/*
rand_next — Generates the next pseudo-random number using the LCG algorithm.

Inputs:
  None

Return:
  int — next pseudo-random number in the range [0, 32767]

Method:
  Updates the global seed using the linear congruential generator formula:
    g_seed = (g_seed * 25173 + 13849) & 0x7FFF
*/
int rand_next(void)
{
    /* PDP-11: LCG with seed*25173 + 13849, keep 15 bits.
       Use long arithmetic: 32767 * 25173 = 824M overflows 16-bit int. */
    g_seed = (int)(((long)g_seed * 25173L + 13849L) & 0x7FFF);
    return g_seed;
}

/*
initw_one — Initialize a single weight group with random Q16 values.

Inputs:
  hi    — pointer to array of high 16-bit words of Q16 weight pairs
  lo    — pointer to array of low 16-bit words of Q16 weight pairs
  n     — number of weights to initialize

Return:
  None

Method:
  For each weight index, generates a random Q8 value in the range [-128, 127]
  using rand_next(), then converts it to Q16 format by splitting into hi/lo
  pairs. The high word is set via sign extension (0 or -1) and the low word
  is set to Q8 << 8. This produces random weights suitable for neural network
  initialization.
*/
static void initw_one(int16_t *hi, int16_t *lo, int n)
{
    int i;
    int r;
    int8_t q8_val;
    for (i = 0; i < n; i++) {
        r = rand_next();
        /* Keep 8 bits [0, 255], then [-128, 127] in Q8 */
        q8_val = (int8_t)(r & 0xFF) - 128;
        /* Convert Q8 -> Q16: Q16 = Q8 << 8 (ashc $8 in PDP-11) */
        hi[i] = (int16_t)(q8_val >> 8);       /* sign extension: 0 or -1 */
        lo[i] = (int16_t)((int16_t)q8_val << 8); /* low word = Q8 << 8 */
    }
}

/*
initw — Initialize all neural network weights with random Q16 values.

Inputs:
  None

Return:
  None

Method:
  Calls initw_one for each of the six weight groups (tok_emb, pos_emb, wq,
  wk, wv, wot), filling each with random Q16 values in the range [-128, 127]
  converted to Q16 format. This provides the initial random weights for the
  neural network before training begins.
*/
void initw(void)
{
    /* PDP-11: Fill all weight groups with random Q16 values.
                Random Q8 in [-128, 127], convert to Q16 by << 8. */
    initw_one(g_q16w.tok_emb.hi, g_q16w.tok_emb.lo, TOK_EMB_N);
    initw_one(g_q16w.pos_emb.hi, g_q16w.pos_emb.lo, POS_EMB_N);
    initw_one(g_q16w.wq.hi,      g_q16w.wq.lo,      WQ_N);
    initw_one(g_q16w.wk.hi,      g_q16w.wk.lo,      WK_N);
    initw_one(g_q16w.wv.hi,      g_q16w.wv.lo,      WV_N);
    initw_one(g_q16w.wot.hi,     g_q16w.wot.lo,     WOT_N);
}

/*
updat_one — Update a single weight group using SGD with the given gradients.

Inputs:
  hi       — pointer to array of high 16-bit words of Q16 weight pairs
  lo       — pointer to array of low 16-bit words of Q16 weight pairs
  grad     — pointer to Q15 gradient array (one gradient per weight)
  n        — number of weights to update
  lr_shift — learning rate shift value (controls effective learning rate)

Return:
  None

Method:
  For each weight, reads the gradient, computes delta = grad >> (lr_shift - 1),
  reconstructs the 32-bit Q16 weight from its hi/lo pair, subtracts delta, and
  stores the updated weight back. The gradient is then zeroed. This implements
  one step of stochastic gradient descent for a single weight group.
*/
static void updat_one(int16_t *hi, int16_t *lo, q15 *grad, int n, int lr_shift)
{
    int i;
    int shift = lr_shift - 1;
    int32_t grad32;
    int32_t delta;
    int32_t w32;
    for (i = 0; i < n; i++) {
        /* Read single Q15 gradient word and zero it (one per weight, per PDP-11 up_do) */
        grad32 = (int32_t)grad[i];
        grad[i] = 0;
        /* delta = grad >> (lr_shift - 1) */
        delta = grad32 >> shift;
        /* Reconstruct 32-bit weight, subtract delta, store back. */
        w32 = (int32_t)(((uint32_t)(uint16_t)hi[i] << 16)
                       | (uint32_t)(uint16_t)lo[i]);
        w32 -= delta;
        hi[i] = (int16_t)(w32 >> 16);
        lo[i] = (int16_t)(uint16_t)w32;
    }
}

/*
updat — Update all neural network weight groups using SGD with the given gradients.

Inputs:
  None

Return:
  None

Method:
  Calls updat_one for each of the six weight groups (tok_emb, pos_emb, wq,
  wk, wv, wot), applying the corresponding gradients and learning rate shifts.
  This performs one step of stochastic gradient descent for the entire network.
*/
void updat(void)
{
    /* PDP-11: For each weight group:
                w_q16 -= grad_q15 >> (lr_shift - 1)
                Then zero gradients.
                Learning rate shifts: tok_emb=4, pos_emb=4, Wq/Wk/Wv=1, Wout=6 */
    updat_one(g_q16w.tok_emb.hi, g_q16w.tok_emb.lo, g_grad.dtke, TOK_EMB_N, 4);
    updat_one(g_q16w.pos_emb.hi, g_q16w.pos_emb.lo, g_grad.dpse, POS_EMB_N, 4);
    updat_one(g_q16w.wq.hi,      g_q16w.wq.lo,      g_grad.dwq,  WQ_N,      1);
    updat_one(g_q16w.wk.hi,      g_q16w.wk.lo,      g_grad.dwk,  WK_N,      1);
    updat_one(g_q16w.wv.hi,      g_q16w.wv.lo,      g_grad.dwv,  WV_N,      1);
    updat_one(g_q16w.wot.hi,     g_q16w.wot.lo,     g_grad.dwot, WOT_N,     6);
}

/*
zerog — Zero all gradient arrays (one Q15 word per weight).

Inputs:
  None

Return:
  None

Method:
  Calls vclr for each gradient array, setting all elements to zero.
*/
void zerog(void)
{
    /* PDP-11: Zero all gradient arrays (one Q15 word per weight) */
    vclr(g_grad.dtke, TOK_EMB_N);
    vclr(g_grad.dpse, POS_EMB_N);
    vclr(g_grad.dwq,  WQ_N);
    vclr(g_grad.dwk,  WK_N);
    vclr(g_grad.dwv,  WV_N);
    vclr(g_grad.dwot, WOT_N);
}

/* ===== TRAIN.MAC — Training loop ===== */

/*
gensm — Generate a random digit sequence and its reverse as training data.

Inputs:
  None

Return:
  None

Method:
  Generates SEQ_LEN random digits in the range [0, 9] using rand_next() and
  stores them in g_trn.tokens. Then reverses the sequence into g_trn.target.
  This creates training pairs where the network must learn to predict the
  reverse of a randomly generated digit sequence.
*/
void gensm(void)
{
    /* PDP-11: Generate 8 random digits [0-9], reverse into target */
    int i = 0;
    for (i = 0; i < SEQ_LEN; i++) {
        int r = rand_next();
        g_trn.tokens[i] = r % 10;
    }
    /* Reverse into target */
    for (i = 0; i < SEQ_LEN; i++)
        g_trn.target[i] = g_trn.tokens[SEQ_LEN - 1 - i];
}

/*
count — Count the number of correct predictions for the current training sample.

Inputs:
  None

Return:
  None

Method:
  Compares the argmax of the logits for each position with the target token.
  Increments the hit and total counters in the training state.
*/
void count(void)
{
    /* PDP-11: Compare argmax(logits[i]) with target[i] for each position */
    int i = 0;
    for (i = 0; i < SEQ_LEN; i++) {
        int max_idx;
        vmax(g_fwd.logits + i * VOCAB, VOCAB, &max_idx);
        if (max_idx == g_trn.target[i])
            g_trn.tr_hit++;
        g_trn.tr_tot++;
    }
}

/*
closs — Compute cross-entropy loss for current sample.

Inputs:
  None

Return:
  Average per-position loss in Q12.

Method:
  Computes the softmax of the logits for each position, looks up the negative
  log-likelihood for the target token, and accumulates the loss. Returns the
  average loss over the sequence.
*/
q12 closs(void)
{
    /* PDP-11: Compute cross-entropy loss for current sample.
                Returns average per-position loss in Q12. */
    int32_t smh = 0, sml = 0;
    int i = 0;
    int j = 0;
    int target;
    int softmax_val;
    q12 loss_q12;
    int32_t result;

    for (i = 0; i < SEQ_LEN; i++) {
        /* Copy logits[i] to dl */
        for (j = 0; j < VOCAB; j++)
            g_bkw.dl[j] = (q8)g_fwd.logits[i * VOCAB + j];

        /* Softmax(DL) in-place */
        sftmx(g_bkw.dl, VOCAB);

        /* Look up -ln(softmax[target[i]]) */
        target = g_trn.target[i];
        softmax_val = (int16_t)g_bkw.dl[target];

        if (softmax_val >= 256)
            continue;  /* p=1.0 -> loss=0 */

        /* Use softmax_val directly as the array index. The original code used 
           softmax_val*2 (a PDP-11 word-offset artifact) which goes out of bounds 
           for softmax_val > 128. */
        loss_q12 = logtbl[softmax_val];

        /* 32-bit accumulate */
        sml += (int32_t)loss_q12;
        smh += (loss_q12 >> 16);
    }

    /* Average: 32-bit sum / 8 (ASHC #-3) */
    result = (smh << 8) + (sml >> 8);
    result >>= 3;
    return (q12)result;
}

/* ===== I/O helpers ===== */

/*
putc_char — Output a single character to stdout.

Inputs:
  c  — character to output

Return:
  None

Method:
  Wraps the standard library putchar() function to output a single character
  to the console. Used throughout the program for formatted output.
*/
void putc_char(char c)
{
    putchar(c);
}

/*
puts_str — Output a null-terminated string to stdout.

Inputs:
  s  — pointer to null-terminated string to output

Return:
  None

Method:
  Iterates through the string character by character, calling putchar() for
  each character until the null terminator is reached. Used throughout the
  program for outputting multi-character strings.
*/
void puts_str(const char *s)
{
    while (*s) putchar(*s++);
}

/*
newln — Output a newline character to stdout.

Inputs:
  None

Return:
  None

Method:
  Calls putchar() with the newline character. Used to move to the next line
  in console output.
*/
void newln(void)
{
    putchar('\n');
}

/*
putoct — Output an integer as a 6-digit octal number.

Inputs:
  val  — integer value to output

Return:
  None

Method:
  Converts the integer to octal representation and outputs each digit using
  putc_char().
*/
void putoct(int val)
{
    /* Print val as 6-digit octal */
    int buf[6];
    int i = 0;
    for (i = 0; i < 6; i++) {
        buf[i] = (val & 07) + '0';
        val >>= 3;
    }
    for (i = 5; i >= 0; i--)
        putc_char((char)buf[i]);
}

/*
putdec — Output an integer as a decimal number.

Inputs:
  val  — integer value to output

Return:
  None

Method:
  Converts the integer to decimal representation and outputs each digit using
  putc_char().
*/
void putdec(int val)
{
    int buf[12], n = 0;
    int i = 0;

    if (val < 0) {
        putc_char('-');
        val = -val;
    }
    if (val == 0) {
        putc_char('0');
        return;
    }
    while (val > 0) {
        buf[n++] = (val % 10) + '0';
        val /= 10;
    }

    for (i = n - 1; i >= 0; i--)
        putc_char((char)buf[i]);
}

/*
putq8 — Output a Q8 fixed-point number as a decimal string.

Inputs:
  val  — Q8 fixed-point value to output

Return:
  None

Method:
  Converts the Q8 fixed-point number to a decimal string with 3 fractional
  digits and outputs each character using putc_char().
*/
void putq8(q8 val)
{
    int32_t frac;
    int d1, d2, d3;

    if ((int16_t)val < 0) {
        putc_char('-');
        val = (q8)(-(int16_t)val);
    }
    /* Integer part */
    putdec((int16_t)val >> 8);
    putc_char('.');
    /* Fractional part (3 digits).
       Use int32_t: max frac = 255 * 1000 = 255000, overflows 16-bit int. */
    frac = (int32_t)((int16_t)val & 0xFF) * 1000L;
    d1 = (int)((frac / 256L) / 100L);
    d2 = (int)(((frac / 256L) % 100L) / 10L);
    d3 = (int)((frac / 256L) % 10L);
    putc_char((char)(d1 + '0'));
    putc_char((char)(d2 + '0'));
    putc_char((char)(d3 + '0'));
}

/*
putspc — Output a space character to stdout.

Inputs:
  None

Return:
  None

Method:
  Calls putc_char() with a space character. Used to separate output elements.
*/
void putspc(void)
{
    putc_char(' ');
}

/*
putvec — Output a vector of Q8 fixed-point numbers as a formatted string.

Inputs:
  vec  — pointer to array of Q8 fixed-point numbers
  len  — length of the vector

Return:
  None

Method:
  Outputs the vector in the format "[x.x, x.x, ...]" using putq8() for each element.
*/
void putvec(const q8 *vec, int len)
{
    int i = 0;

    putc_char('[');
    for (i = 0; i < len; i++) {
        putq8(vec[i]);
        if (i < len - 1) {
            putc_char(',');
            putspc();
        }
    }
    putc_char(']');
}

/*
report — Output the current training step, loss, and accuracy.

Inputs:
  None

Return:
  None

Method:
  Outputs the current training step, loss, and accuracy in a formatted string.
*/
void report(void)
{
    /* PDP-11: Print " step N loss=X.XXXX accuracy=X.XXX" */
    int step;
    q12 loss;
    int32_t frac;
    int32_t permille;

    puts_str(" step ");
    step = g_trn.tr_stp;
    if (step < 1000) putc_char(' ');
    if (step < 100) putc_char(' ');
    putdec(step);

    puts_str(" loss=");
    loss = closs();
    putdec((int16_t)(loss >> 12));
    putc_char('.');
    /* Use int32_t: max frac = 4095 * 10000 = 40,950,000, overflows 16-bit int. */
    frac = ((int32_t)loss & 0xFFF) * 10000L / 4096L;
    putc_char((char)(frac / 1000L + '0'));
    putc_char((char)((frac / 100L) % 10L + '0'));
    putc_char((char)((frac / 10L) % 10L + '0'));
    putc_char((char)(frac % 10L + '0'));

    puts_str(" accuracy=");
    /* Use int32_t: max tr_hit * 1000 = 400 * 1000 = 400000, overflows 16-bit int. */
    permille = (int32_t)g_trn.tr_hit * 1000L / (int32_t)g_trn.tr_tot;
    if (permille >= 1000) {
        putc_char('1');
        putc_char('.');
        puts_str("000");
    } else {
        putc_char('0');
        putc_char('.');
        putc_char((char)(permille / 100L + '0'));
        putc_char((char)((permille / 10L) % 10L + '0'));
        putc_char((char)(permille % 10L + '0'));
    }
    newln();

    g_trn.tr_hit = 0;
    g_trn.tr_tot = 0;
}

/*
test — Perform a final test on 10 samples and output results.

Inputs:
  None

Return:
  None

Method:
  Generates 10 test samples, performs forward passes, and compares predictions
  to targets. Outputs each sample with OK/FAIL and final accuracy.
*/
void test(void)
{
    /* PDP-11: Final test — 10 samples, print each with OK/FAIL */
    int sok = 0;
    int t = 0;
    int pred[SEQ_LEN];
    int i = 0;
    int max_idx;
    int all_ok;

    for (t = 0; t < 10; t++) {
        gensm();
        cvt16();
        forwrd();

        /* Store predictions */
        for (i = 0; i < SEQ_LEN; i++) {
            vmax(g_fwd.logits + i * VOCAB, VOCAB, &max_idx);
            pred[i] = max_idx;
        }

        /* Print: " i i i i i i i i -> p p p p p p p p  OK/FAIL" */
        putspc();
        for (i = 0; i < SEQ_LEN; i++) {
            putc_char((char)(g_trn.tokens[i] + '0'));
            putspc();
        }
        puts_str("-> ");
        for (i = 0; i < SEQ_LEN; i++) {
            putc_char((char)(pred[i] + '0'));
            putspc();
        }

        /* Check all correct */
        all_ok = 1;
        for (i = 0; i < SEQ_LEN; i++) {
            if (g_trn.target[i] != pred[i]) {
                all_ok = 0;
                break;
            }
        }
        if (all_ok) {
            sok++;
            puts_str(" OK");
        } else {
            puts_str(" FAIL");
        }
        newln();
    }

    newln();
    puts_str(" accuracy: ");
    putdec(sok);
    puts_str("/10");
    newln();
}

/*
train — Entry point for the neural network training loop.

Inputs:
  None

Return:
  int — always returns 0

Method:
  Prints a header message, initializes weights with a random seed of 887,
  then runs 351 training steps. Each step generates a random digit sequence,
  performs a forward pass, backward pass, and weight update, then counts
  correct predictions and reports the result. After training completes,
  runs a final test on 10 samples and waits for the user to press RETURN
  before returning to DOS.
*/
int train(void)
{
    /* PDP-11: Training entry point */
    puts_str("attn/atari-paper tape is all you need\n");    
    puts_str("d=16 seq=8 v=10 params=1216 q8/q15/q16\n");
    newln();
    puts_str("training...\n");

    /* Initialize weights */
    rand_seed(887);
    initw();

    /* Training loop */
    g_trn.tr_stp = 1;
    g_trn.tr_hit = 0;
    g_trn.tr_tot = 0;

    for (; g_trn.tr_stp <= 351; g_trn.tr_stp++) {
        /* One training step */
        gensm();
        cvt16();
        forwrd();
        bkwrd();
        updat();

        /* Count correct */
        count();

        /* Report every step */
        report();
    }

    /* Final test */
    newln();
    puts_str("test\n");
    test();

    puts_str("press RETURN to exit program\n");
    getchar();

    return(0);
}

/*
main — Program entry point: calls train() and returns to Atari DOS.

Inputs:
  argc  — argument count (ignored)
  argv  — argument vector (ignored)

Return:
  int — always returns 0

Method:
  Calls train() to execute the full training and testing workflow, then
  jumps to address 0x000A (DOSVEC) to return control to the Atari operating
  system. The argc and argv parameters are cast to void to suppress
  unused-parameter warnings.
*/
int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    train();
    (*(void (**)()) 0x000A)();  /* return to DOS via DOSVEC */
    return 0;
}