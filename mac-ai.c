#include "mac-ai.h"

static int g_seed = 887;  /* PDP-11 seed value */

/* ===== Lookup tables ===== */
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
q8 fxdiv(q8 dividend, q8 divisor)
{
    if (divisor == 0) return 0;
    int32_t num = (int32_t)dividend << 8;
    return (q8)(num / (int32_t)divisor);
}

q16 fxmul(q8 a, q8 b)
{
    /* PDP-11: mul b, a  -> R0:R1 = a*b (Q24)
                ashc $-8, r0  -> R0 = Q8 result (high word) */
    int32_t product = (int32_t)a * (int32_t)b;  /* Q24 */
    return (q16)(product >> 8);                   /* Q8 in high word */
}

/* ===== VECOP.MAC — Level 1 vector operations ===== */

q8 vdot(const q8 *x, const q8 *y, int len)
{
    /* PDP-11: dthi:dtlo are 16-bit registers (mp_ahi/dthi, mp_alo/dtlo).
                add r1, dtlo  (16-bit unsigned add, sets carry)
                adc dthi      (add carry)
                add r0, dthi  (add high word of product)
                ashc $-8, r0  (shift 32-bit pair right 8, clamp to int16) */
    uint16_t dtlo = 0;
    uint16_t dthi = 0;

    for (int i = 0; i < len; i++) {
        int32_t prod = (int32_t)x[i] * (int32_t)y[i];
        uint32_t lo_sum = (uint32_t)dtlo + (uint32_t)(uint16_t)prod;
        uint16_t carry = (uint16_t)(lo_sum >> 16);
        dtlo = (uint16_t)lo_sum;
        dthi = (uint16_t)(dthi + carry + (uint16_t)(int16_t)(prod >> 16));
    }

    int32_t acc = ((int32_t)(int16_t)dthi << 16) | (uint32_t)dtlo;
    int32_t result = acc >> 8;
    if (result > 32767)  result = 32767;
    if (result < -32768) result = -32768;
    return (q8)result;
}

void vadd(const q8 *x, const q8 *y, q8 *z, int len)
{
    /* PDP-11: z[i] = x[i]; z[i] += y[i] */
    for (int i = 0; i < len; i++)
        z[i] = (q8)((int16_t)x[i] + (int16_t)y[i]);
}

void vsub(const q8 *x, const q8 *y, q8 *z, int len)
{
    /* PDP-11: z[i] = x[i]; z[i] -= y[i] */
    for (int i = 0; i < len; i++)
        z[i] = (q8)((int16_t)x[i] - (int16_t)y[i]);
}

void vscl(const q8 *x, q8 *y, int len, q8 alpha)
{
    /* PDP-11: y[i] = (alpha * x[i]) >> 8 */
    for (int i = 0; i < len; i++) {
        int32_t prod = (int32_t)alpha * (int32_t)x[i];
        y[i] = (q8)(prod >> 8);
    }
}

int vmax(const q8 *vec, int len, int *out_idx)
{
    /* PDP-11: R0=max value, R1=index of max */
    int max_val = (int16_t)vec[0];
    int max_idx = 0;

    for (int i = 1; i < len; i++) {
        int val = (int16_t)vec[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    if (out_idx) *out_idx = max_idx;
    return max_val;
}

void vcpy(const q8 *src, q8 *dst, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = src[i];
}

void vclr(q8 *vec, int len)
{
    for (int i = 0; i < len; i++)
        vec[i] = 0;
}

void vsadd(const q8 *src, q8 *dst, int len, q8 scalar)
{
    /* PDP-11: dst[k] += (scalar * src[k]) >> 8, with clamp */
    for (int i = 0; i < len; i++) {
        int32_t prod = (int32_t)scalar * (int32_t)src[i];
        int32_t val = prod >> 8;

        /* Clamp */
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;

        int32_t sum = (int16_t)dst[i] + val;
        if (sum > 32767) sum = 32767;
        if (sum < -32768) sum = -32768;
        dst[i] = (q8)sum;
    }
}

/* ===== MATOP.MAC — Level 2 matrix-vector operations ===== */

void mvmul(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols)
{
    /* PDP-11: vout[i] = sum_j(mat[i][j] * vin[j]), 16-bit dthi:dtlo accumulator,
                then ashc $-8, clamp (matches mp_ahi/mp_alo in MATOP.MAC). */
    for (int i = 0; i < rows; i++) {
        uint16_t dtlo = 0;
        uint16_t dthi = 0;

        for (int j = 0; j < cols; j++) {
            int32_t prod = (int32_t)mat[i * cols + j] * (int32_t)vin[j];
            uint32_t lo_sum = (uint32_t)dtlo + (uint32_t)(uint16_t)prod;
            uint16_t carry = (uint16_t)(lo_sum >> 16);
            dtlo = (uint16_t)lo_sum;
            dthi = (uint16_t)(dthi + carry + (uint16_t)(int16_t)(prod >> 16));
        }

        int32_t acc = ((int32_t)(int16_t)dthi << 16) | (uint32_t)dtlo;
        int32_t result = acc >> 8;
        if (result > 32767)  result = 32767;
        if (result < -32768) result = -32768;
        vout[i] = (q8)result;
    }
}

void mvadd(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols)
{
    /* PDP-11: vout[i] += sum_j(mat[i][j] * vin[j]), 16-bit dthi:dtlo accumulator,
                then ashc $-8, clamp inner, add to vout[i] with clamp (MVADD.MAC). */
    for (int i = 0; i < rows; i++) {
        uint16_t dtlo = 0;
        uint16_t dthi = 0;

        for (int j = 0; j < cols; j++) {
            int32_t prod = (int32_t)mat[i * cols + j] * (int32_t)vin[j];
            uint32_t lo_sum = (uint32_t)dtlo + (uint32_t)(uint16_t)prod;
            uint16_t carry = (uint16_t)(lo_sum >> 16);
            dtlo = (uint16_t)lo_sum;
            dthi = (uint16_t)(dthi + carry + (uint16_t)(int16_t)(prod >> 16));
        }

        int32_t acc = ((int32_t)(int16_t)dthi << 16) | (uint32_t)dtlo;
        int32_t result = acc >> 8;
        if (result > 32767)  result = 32767;
        if (result < -32768) result = -32768;

        int32_t sum = (int16_t)vout[i] + result;
        if (sum > 32767)  sum = 32767;
        if (sum < -32768) sum = -32768;
        vout[i] = (q8)sum;
    }
}

void vtmul(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols)
{
    /* PDP-11: vout[j] = sum_i(mat[i][j] * vin[i]), per-element Q8 rounding.
                Clears vout first, then accumulates row by row. */
    /* Clear output */
    for (int j = 0; j < cols; j++)
        vout[j] = 0;

    /* Accumulate row by row */
    for (int i = 0; i < rows; i++) {
        q8 scalar = vin[i];
        for (int j = 0; j < cols; j++) {
            int32_t prod = (int32_t)mat[i * cols + j] * (int32_t)scalar;
            int32_t val = prod >> 8;

            /* Clamp the product */
            if (val > 32767) val = 32767;
            if (val < -32768) val = -32768;

            int32_t sum = (int16_t)vout[j] + val;
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            vout[j] = (q8)sum;
        }
    }
}

void outer(const q8 *vx, const q8 *vy, q8 *mat, int rows, int cols)
{
    /* PDP-11: mat[i][j] += vx[i] * vy[j] >> 8, with clamp.
                For backward pass: weight gradient accumulation. */
    for (int i = 0; i < rows; i++) {
        q8 scalar = vx[i];
        for (int j = 0; j < cols; j++) {
            int32_t prod = (int32_t)scalar * (int32_t)vy[j];
            int32_t val = prod >> 8;

            /* Clamp the product */
            if (val > 32767) val = 32767;
            if (val < -32768) val = -32768;

            int32_t sum = (int16_t)mat[i * cols + j] + val;
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            mat[i * cols + j] = (q8)sum;
        }
    }
}

/* ===== ACTFN.MAC — Activation functions ===== */

void vrelu(q8 *vec, int len)
{
    /* PDP-11: negative elements zeroed, positive kept */
    for (int i = 0; i < len; i++)
        if ((int16_t)vec[i] < 0)
            vec[i] = 0;
}

void sftmx(q8 *vec, int len)
{
    /* PDP-11: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
                Uses EXPTBL lookup for exp(), FXDIV for division. */

    /* Step 1: Find max value */
    int max_val = (int16_t)vec[0];
    for (int i = 1; i < len; i++) {
        int val = (int16_t)vec[i];
        if (val > max_val)
            max_val = val;
    }

    /* Step 2: exp(x_i - max) via LUT, accumulate sum */
    int32_t sum = 0;
    for (int i = 0; i < len; i++) {
        int diff = (int16_t)vec[i] - max_val;  /* <= 0 */
        int idx = (-diff) >> 3;                  /* divide by 8 -> table index */
        if (idx > 255) idx = 255;                /* clamp to table bounds */
        int exp_val = exptbl[idx];
        vec[i] = (q8)exp_val;                    /* store in-place */
        sum += exp_val;
    }

    /* Step 3: Normalize — divide each exp value by sum */
    for (int i = 0; i < len; i++)
        vec[i] = fxdiv(vec[i], (q8)sum);
}

/* ===== LAYER.MAC — Neural network layers ===== */

void embed(const int *tokens, const q8 *tkemb, const q8 *psemb,
           q8 *xout, int seq_len, int d_model)
{
    /* PDP-11: output[i] = token_embed[tokens[i]] + pos_embed[i]
                Row size = d_model * 2 bytes */
    for (int i = 0; i < seq_len; i++) {
        int tok_id = tokens[i];
        const q8 *tok_row = tkemb + tok_id * d_model;
        q8 *out_row = xout + i * d_model;

        for (int j = 0; j < d_model; j++)
            out_row[j] = (q8)((int16_t)tok_row[j] + (int16_t)psemb[i * d_model + j]);
    }
}

void proj(const q8 *yin, const q8 *wout, q8 *logits,
          int seq_len, int d_model, int vocab)
{
    /* PDP-11: logits[i] = Wout^T * Y[i] for each position.
                Uses self-modifying code to patch VTMUL params. */
    for (int i = 0; i < seq_len; i++) {
        vtmul(wout, yin, logits, d_model, vocab);
        yin += d_model;
        logits += vocab;
    }
}

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

    /* Steps 1-3: Q=X.Wq, K=X.Wk, V=X.Wv — per-position vtmul loop
       PDP-11 at_bpr: loops seq_len times calling vtmul(W, X[s], out[s], d, d) */
    for (int s = 0; s < seq_len; s++) {
        vtmul(wq, xin + s * d_model, Q + s * d_model, d_model, d_model);
        vtmul(wk, xin + s * d_model, K + s * d_model, d_model, d_model);
        vtmul(wv, xin + s * d_model, V + s * d_model, d_model, d_model);
    }

    /* Step 4: S[i][j] = Q[i].K[j] / sqrt(d) */
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            q8 dot = vdot(Q + i * d_model, K + j * d_model, d_model);
            dot = (q8)((int16_t)dot >> (-shift));  /* divide by sqrt(d) */
            S[i * seq_len + j] = dot;
        }
    }

    /* Step 5: softmax per row */
    for (int i = 0; i < seq_len; i++)
        sftmx(S + i * seq_len, seq_len);

    /* Step 6: Y[i] = V^T . S[i] */
    for (int i = 0; i < seq_len; i++)
        vtmul(V, S + i * seq_len, yout + i * d_model, seq_len, d_model);

    /* Step 7: Y += X (residual) */
    for (int i = 0; i < seq_len * d_model; i++)
        yout[i] = (q8)((int16_t)yout[i] + (int16_t)xin[i]);
}

/* ===== FORWRD.MAC — Forward pass & weight conversion ===== */

void cvt16(void)
{
    /* PDP-11: Convert Q16 weights (hi/lo pairs) to Q8 for forward/backward.
                For each weight: Q16 >> 8 = Q8. */

    /* tok_emb */
    for (int i = 0; i < TOK_EMB_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.tok_emb.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.tok_emb.lo[i]);
        g_q8w.tok_emb[i] = (q8)(q16 >> 8);
    }
    /* pos_emb */
    for (int i = 0; i < POS_EMB_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.pos_emb.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.pos_emb.lo[i]);
        g_q8w.pos_emb[i] = (q8)(q16 >> 8);
    }
    /* Wq */
    for (int i = 0; i < WQ_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wq.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wq.lo[i]);
        g_q8w.wq[i] = (q8)(q16 >> 8);
    }
    /* Wk */
    for (int i = 0; i < WK_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wk.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wk.lo[i]);
        g_q8w.wk[i] = (q8)(q16 >> 8);
    }
    /* Wv */
    for (int i = 0; i < WV_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wv.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wv.lo[i]);
        g_q8w.wv[i] = (q8)(q16 >> 8);
    }
    /* Wout */
    for (int i = 0; i < WOT_N; i++) {
        int32_t q16 = (int32_t)(((uint32_t)(uint16_t)g_q16w.wot.hi[i] << 16)
                               | (uint32_t)(uint16_t)g_q16w.wot.lo[i]);
        g_q8w.wot[i] = (q8)(q16 >> 8);
    }
}

void forwrd(void)
{
    /* PDP-11: 1. Embedding, 2. Self-attention, 3. Output projection */
    embed(g_trn.tokens, g_q8w.tok_emb, g_q8w.pos_emb, g_fwd.xx, SEQ_LEN, D_MODEL);
    attn(g_fwd.xx, g_q8w.wq, g_q8w.wk, g_q8w.wv, g_fwd.yy, g_fwd.work,
         SEQ_LEN, D_MODEL, 2);  /* sqrtsh=2 for d_model=16 */
    proj(g_fwd.yy, g_q8w.wot, g_fwd.logits, SEQ_LEN, D_MODEL, VOCAB);
}

/* ===== BKWRD.MAC — Backward pass ===== */
void bkwrd(void)
{
    /* Workspace pointers into ATTN WORK area */
    q8 *Q = g_fwd.work;
    q8 *K = Q + SEQ_LEN * D_MODEL;
    q8 *V = K + SEQ_LEN * D_MODEL;
    q8 *A = V + SEQ_LEN * D_MODEL;  /* post-softmax attention */

    /* === Step 1: dLogits, dWout, dY === */

    /* Clear dY */
    vclr(g_bkw.dy, SEQ_LEN * D_MODEL);

    for (int i = 0; i < SEQ_LEN; i++) {
        /* Copy logits[i] to dl */
        for (int j = 0; j < VOCAB; j++)
            g_bkw.dl[j] = g_fwd.logits[i * VOCAB + j];

        /* Softmax(DL) in-place */
        sftmx(g_bkw.dl, VOCAB);

        /* DL[target[i]] -= 256 (1.0 Q8) */
        g_bkw.dl[g_trn.target[i]] = (q8)((int16_t)g_bkw.dl[g_trn.target[i]] - 256);

        /* Shift DL to Q15: << 7 */
        for (int j = 0; j < VOCAB; j++)
            g_bkw.dl[j] = (q15)((int16_t)g_bkw.dl[j] << 7);

        /* dWout += OUTER(Y[i], DL, D_MODEL, VOCAB) — assembly bk_p4: 16. */
        outer(g_fwd.yy + i * D_MODEL, g_bkw.dl, g_grad.dwot, D_MODEL, VOCAB);

        /* dY[i] = MVMUL(Wout, DL) */
        mvmul(g_q8w.wot, g_bkw.dl, g_bkw.dy + i * D_MODEL, D_MODEL, VOCAB);
    }

    /* === Step 2: Backward O = A.V -> dA, dV === */

    /* Clear dV */
    vclr(g_bkw.dvv, SEQ_LEN * D_MODEL);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            /* dA[i][j] = VDOT(V[j], dY[i], D) */
            q15 dot = (q15)vdot(V + j * D_MODEL, g_bkw.dy + i * D_MODEL, D_MODEL);
            g_bkw.da[i * SEQ_LEN + j] = dot;

            /* dV[j][k] += A[i][j] * dY[i][k] >> 8 (VSADD) */
            q8 a_ij = A[i * SEQ_LEN + j];
            vsadd(g_bkw.dy + i * D_MODEL, g_bkw.dvv + j * D_MODEL, D_MODEL, a_ij);
        }
    }

    /* === Step 3: Backward softmax -> dSc (stored in DA) === */

    for (int i = 0; i < SEQ_LEN; i++) {
        /* dot_ad = VDOT(A[i], dA[i], S) */
        q15 dot_ad = (q15)vdot(A + i * SEQ_LEN, g_bkw.da + i * SEQ_LEN, SEQ_LEN);

        /* dSc[i][j] = A[i][j] * (dA[i][j] - dot_ad) >> 8, then >> sqrt_shift */
        for (int j = 0; j < SEQ_LEN; j++) {
            q15 da_val = g_bkw.da[i * SEQ_LEN + j];
            int32_t diff = (int16_t)da_val - (int16_t)dot_ad;

            /* Clamp */
            if (diff > 32767) diff = 32767;
            if (diff < -32768) diff = -32768;

            int32_t prod = (int32_t)A[i * SEQ_LEN + j] * diff;
            int32_t val = prod >> 8;

            /* >> sqrt_shift (scaling, asm uses ash $-2 for d_model=16) */
            val >>= 2;

            g_bkw.da[i * SEQ_LEN + j] = (q15)val;
        }
    }

    /* === Step 4: Backward Q.K^T -> dQ, dK === */

    /* dQ[i] = K^T . dSc[i] */
    for (int i = 0; i < SEQ_LEN; i++)
        vtmul(K, g_bkw.da + i * SEQ_LEN, g_bkw.dqq + i * D_MODEL, SEQ_LEN, D_MODEL);

    /* dK: for each j, extract column j of dSc, then VTMUL */
    for (int j = 0; j < SEQ_LEN; j++) {
        /* Extract column j from DA into dtmp */
        for (int i = 0; i < SEQ_LEN; i++)
            g_bkw.dtmp[i] = g_bkw.da[i * SEQ_LEN + j];

        vtmul(Q, g_bkw.dtmp, g_bkw.dkk + j * D_MODEL, SEQ_LEN, D_MODEL);
    }

    /* === Step 5: Backward projections + dX === */

    /* Copy dY -> dX */
    vcpy(g_bkw.dy, g_bkw.dxx, SEQ_LEN * D_MODEL);

    for (int i = 0; i < SEQ_LEN; i++) {
        /* dX[i] += Wq * dQ[i] */
        mvadd(g_q8w.wq, g_bkw.dqq + i * D_MODEL, g_bkw.dxx + i * D_MODEL, D_MODEL, D_MODEL);
        /* dWq += OUTER(X[i], dQ[i]) */
        outer(g_fwd.xx + i * D_MODEL, g_bkw.dqq + i * D_MODEL, g_grad.dwq, D_MODEL, D_MODEL);

        /* dX[i] += Wk * dK[i] */
        mvadd(g_q8w.wk, g_bkw.dkk + i * D_MODEL, g_bkw.dxx + i * D_MODEL, D_MODEL, D_MODEL);
        /* dWk += OUTER(X[i], dK[i]) */
        outer(g_fwd.xx + i * D_MODEL, g_bkw.dkk + i * D_MODEL, g_grad.dwk, D_MODEL, D_MODEL);

        /* dX[i] += Wv * dV[i] */
        mvadd(g_q8w.wv, g_bkw.dvv + i * D_MODEL, g_bkw.dxx + i * D_MODEL, D_MODEL, D_MODEL);
        /* dWv += OUTER(X[i], dV[i]) */
        outer(g_fwd.xx + i * D_MODEL, g_bkw.dvv + i * D_MODEL, g_grad.dwv, D_MODEL, D_MODEL);
    }

    /* === Step 6: Backward embedding === */

    for (int i = 0; i < SEQ_LEN; i++) {
        /* d_tok[token[i]] += dX[i] */
        int tok = g_trn.tokens[i];
        for (int k = 0; k < D_MODEL; k++) {
            int32_t sum = (int16_t)g_grad.dtke[tok * D_MODEL + k] + (int16_t)g_bkw.dxx[i * D_MODEL + k];
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            g_grad.dtke[tok * D_MODEL + k] = (q15)sum;
        }

        /* d_pos[i] += dX[i] */
        for (int k = 0; k < D_MODEL; k++) {
            int32_t sum = (int16_t)g_grad.dpse[i * D_MODEL + k] + (int16_t)g_bkw.dxx[i * D_MODEL + k];
            if (sum > 32767) sum = 32767;
            if (sum < -32768) sum = -32768;
            g_grad.dpse[i * D_MODEL + k] = (q15)sum;
        }
    }
}

/* ===== UPDAT.MAC — Weight init, SGD update, gradient zeroing ===== */
void rand_seed(int seed)
{
    g_seed = seed;
}

int rand_next(void)
{
    /* PDP-11: LCG with seed*25173 + 13849, keep 15 bits */
    g_seed = (g_seed * 25173 + 13849) & 0x7FFF;
    return g_seed;
}

void initw(void)
{
    /* PDP-11: Fill all weight groups with random Q16 values.
                Random Q8 in [-128, 127], convert to Q16 by << 8. */

    struct { int16_t *hi; int16_t *lo; int n; } groups[] = {
        { g_q16w.tok_emb.hi, g_q16w.tok_emb.lo, TOK_EMB_N },
        { g_q16w.pos_emb.hi, g_q16w.pos_emb.lo, POS_EMB_N },
        { g_q16w.wq.hi,      g_q16w.wq.lo,      WQ_N },
        { g_q16w.wk.hi,      g_q16w.wk.lo,      WK_N },
        { g_q16w.wv.hi,      g_q16w.wv.lo,      WV_N },
        { g_q16w.wot.hi,     g_q16w.wot.lo,     WOT_N },
    };

    for (int g = 0; g < 6; g++) {
        for (int i = 0; i < groups[g].n; i++) {
            int r = rand_next();
            /* Keep 8 bits [0, 255], then [-128, 127] in Q8 */
            int8_t q8_val = (int8_t)(r & 0xFF) - 128;
            /* Convert Q8 -> Q16: Q16 = Q8 << 8 (ashc $8 in PDP-11) */
            groups[g].hi[i] = (int16_t)(q8_val >> 8);       /* sign extension: 0 or -1 */
            groups[g].lo[i] = (int16_t)((int16_t)q8_val << 8); /* low word = Q8 << 8 */
        }
    }
}

void updat(void)
{
    /* PDP-11: For each weight group:
                w_q16 -= grad_q15 >> (lr_shift - 1)
                Then zero gradients.
                Learning rate shifts: tok_emb=4, pos_emb=4, Wq/Wk/Wv=1, Wout=6 */

    struct {
        int16_t *hi;
        int16_t *lo;
        q15 *grad;
        int n;
        int lr_shift;
    } groups[] = {
        { g_q16w.tok_emb.hi, g_q16w.tok_emb.lo, g_grad.dtke, TOK_EMB_N, 4 },
        { g_q16w.pos_emb.hi, g_q16w.pos_emb.lo, g_grad.dpse, POS_EMB_N, 4 },
        { g_q16w.wq.hi,      g_q16w.wq.lo,      g_grad.dwq,  WQ_N,      1 },
        { g_q16w.wk.hi,      g_q16w.wk.lo,      g_grad.dwk,  WK_N,      1 },
        { g_q16w.wv.hi,      g_q16w.wv.lo,      g_grad.dwv,  WV_N,      1 },
        { g_q16w.wot.hi,     g_q16w.wot.lo,     g_grad.dwot, WOT_N,     6 },
    };

    for (int g = 0; g < 6; g++) {
        int shift = groups[g].lr_shift - 1;  /* shift-1 */
        for (int i = 0; i < groups[g].n; i++) {
            /* Read single Q15 gradient word and zero it (one per weight, per PDP-11 up_do) */
            int32_t grad32 = (int32_t)groups[g].grad[i];
            groups[g].grad[i] = 0;

            /* delta = grad >> (lr_shift - 1) */
            int32_t delta = grad32 >> shift;

            /* Reconstruct 32-bit weight, subtract delta, store back. */
            int32_t w32 = (int32_t)(((uint32_t)(uint16_t)groups[g].hi[i] << 16)
                                   | (uint32_t)(uint16_t)groups[g].lo[i]);
            w32 -= delta;
            groups[g].hi[i] = (int16_t)(w32 >> 16);
            groups[g].lo[i] = (int16_t)(uint16_t)w32;
        }
    }
}

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

/* Global state definitions */
q16_weights g_q16w;
q8_weights  g_q8w;
gradients   g_grad;
forward_cache g_fwd;
backward_workspace g_bkw;
training_state  g_trn;

void gensm(void)
{
    /* PDP-11: Generate 8 random digits [0-9], reverse into target */
    for (int i = 0; i < SEQ_LEN; i++) {
        int r = rand_next();
        g_trn.tokens[i] = r % 10;
    }
    /* Reverse into target */
    for (int i = 0; i < SEQ_LEN; i++)
        g_trn.target[i] = g_trn.tokens[SEQ_LEN - 1 - i];
}

void count(void)
{
    /* PDP-11: Compare argmax(logits[i]) with target[i] for each position */
    for (int i = 0; i < SEQ_LEN; i++) {
        int max_idx;
        vmax(g_fwd.logits + i * VOCAB, VOCAB, &max_idx);
        if (max_idx == g_trn.target[i])
            g_trn.tr_hit++;
        g_trn.tr_tot++;
    }
}

q12 closs(void)
{
    /* PDP-11: Compute cross-entropy loss for current sample.
                Returns average per-position loss in Q12. */
    int32_t smh = 0, sml = 0;

    for (int i = 0; i < SEQ_LEN; i++) {
        /* Copy logits[i] to dl */
        for (int j = 0; j < VOCAB; j++)
            g_bkw.dl[j] = (q8)g_fwd.logits[i * VOCAB + j];

        /* Softmax(DL) in-place */
        sftmx(g_bkw.dl, VOCAB);

        /* Look up -ln(softmax[target[i]]) */
        int target = g_trn.target[i];
        int softmax_val = (int16_t)g_bkw.dl[target];

        if (softmax_val >= 256)
            continue;  /* p=1.0 -> loss=0 */

        /* Use softmax_val directly as the array index. The original code used 
           softmax_val*2 (a PDP-11 word-offset artifact) which goes out of bounds 
           for softmax_val > 128. */
        q12 loss_q12 = logtbl[softmax_val];

        /* 32-bit accumulate */
        sml += (int32_t)loss_q12;
        smh += (loss_q12 >> 16);
    }

    /* Average: 32-bit sum / 8 (ASHC #-3) */
    int32_t result = (smh << 8) + (sml >> 8);
    result >>= 3;
    return (q12)result;
}

/* ===== I/O helpers ===== */

void putc_char(char c)
{
    putchar(c);
}

void puts_str(const char *s)
{
    while (*s) putchar(*s++);
}

void newln(void)
{
    putchar('\n');
}

void putoct(int val)
{
    /* Print val as 6-digit octal */
    int buf[6];
    for (int i = 0; i < 6; i++) {
        buf[i] = (val & 07) + '0';
        val >>= 3;
    }
    for (int i = 5; i >= 0; i--)
        putc_char((char)buf[i]);
}

void putdec(int val)
{
    if (val < 0) {
        putc_char('-');
        val = -val;
    }
    if (val == 0) {
        putc_char('0');
        return;
    }
    int buf[12], n = 0;
    while (val > 0) {
        buf[n++] = (val % 10) + '0';
        val /= 10;
    }
    for (int i = n - 1; i >= 0; i--)
        putc_char((char)buf[i]);
}

void putq8(q8 val)
{
    if ((int16_t)val < 0) {
        putc_char('-');
        val = (q8)(-(int16_t)val);
    }
    /* Integer part */
    putdec((int16_t)val >> 8);
    putc_char('.');
    /* Fractional part (3 digits) */
    int frac = ((int16_t)val & 0xFF) * 1000;
    int d1 = (frac / 256) / 100;
    int d2 = ((frac / 256) % 100) / 10;
    int d3 = (frac / 256) % 10;
    putc_char((char)(d1 + '0'));
    putc_char((char)(d2 + '0'));
    putc_char((char)(d3 + '0'));
}

void putspc(void)
{
    putc_char(' ');
}

void putvec(const q8 *vec, int len)
{
    putc_char('[');
    for (int i = 0; i < len; i++) {
        putq8(vec[i]);
        if (i < len - 1) {
            putc_char(',');
            putspc();
        }
    }
    putc_char(']');
}

void report(void)
{
    /* PDP-11: Print " step N loss=X.XXXX accuracy=X.XXX" */
    puts_str(" step ");
    int step = g_trn.tr_stp;
    if (step < 1000) putc_char(' ');
    if (step < 100) putc_char(' ');
    putdec(step);

    puts_str(" loss=");
    q12 loss = closs();
    putdec((int16_t)(loss >> 12));
    putc_char('.');
    int frac = ((int16_t)loss & 0xFFF) * 10000 / 4096;
    putc_char((char)(frac / 1000 + '0'));
    putc_char((char)((frac / 100) % 10 + '0'));
    putc_char((char)((frac / 10) % 10 + '0'));
    putc_char((char)(frac % 10 + '0'));

    puts_str(" accuracy=");
    int permille = g_trn.tr_hit * 1000 / g_trn.tr_tot;
    if (permille >= 1000) {
        putc_char('1');
        putc_char('.');
        puts_str("000");
    } else {
        putc_char('0');
        putc_char('.');
        putc_char((char)(permille / 100 + '0'));
        putc_char((char)((permille / 10) % 10 + '0'));
        putc_char((char)(permille % 10 + '0'));
    }
    newln();

    g_trn.tr_hit = 0;
    g_trn.tr_tot = 0;
}

void test(void)
{
    /* PDP-11: Final test — 10 samples, print each with OK/FAIL */
    int sok = 0;

    for (int t = 0; t < 10; t++) {
        gensm();
        cvt16();
        forwrd();



        /* Store predictions */
        int pred[SEQ_LEN];
        for (int i = 0; i < SEQ_LEN; i++) {
            int max_idx;
            vmax(g_fwd.logits + i * VOCAB, VOCAB, &max_idx);
            pred[i] = max_idx;
        }

        /* Print: " i i i i i i i i -> p p p p p p p p  OK/FAIL" */
        putspc();
        for (int i = 0; i < SEQ_LEN; i++) {
            putc_char((char)(g_trn.tokens[i] + '0'));
            putspc();
        }
        puts_str("-> ");
        for (int i = 0; i < SEQ_LEN; i++) {
            putc_char((char)(pred[i] + '0'));
            putspc();
        }

        /* Check all correct */
        int all_ok = 1;
        for (int i = 0; i < SEQ_LEN; i++) {
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

int train(void)
{
    /* PDP-11: Training entry point */
    puts_str(" attn/11 — paper tape is all you need.\n");
    puts_str(" d=16 seq=8 v=10 params=1216 q8/q15/q16\n");
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

        /* Report every 50 steps */
        if (g_trn.tr_stp % 50 == 0)
            report();
    }

    /* Final test */
    newln();
    puts_str("test\n");
    test();

    return(0);
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    return train();
}