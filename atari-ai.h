/*
 * atari-ai.h — PDP-11 ATTN/11 → Atari 8-bit C port
 *
 * Fixed-point formats:
 *   Q7.8  = int16_t, value = raw / 256.0
 *   Q15   = int16_t, value = raw / 32768.0  (gradients)
 *   Q16   = int32_t, value = raw / 256.0    (weight accumulators)
 *   Q12   = int32_t, value = raw / 4096.0   (loss table)
 *
 * Network config (hard-coded in the asm):
 *   seq_len = 8, d_model = 16, vocab = 10
 */

#ifndef ATARI_AI_H
#define ATARI_AI_H

#include <stdint.h>
#include <stdio.h>

/* ===== Fixed-point type aliases ===== */
typedef int16_t q8;   /* Q7.8  — forward weights, activations */
typedef int16_t q15;  /* Q15   — gradient accumulators          */
typedef int32_t q16;  /* Q16   — weight accumulators (32-bit)   */
typedef int32_t q12;  /* Q12   — loss table entries             */

/* ===== Network constants ===== */
#define SEQ_LEN   8
#define D_MODEL   16
#define VOCAB     10

#define TOK_EMB_N   (VOCAB * D_MODEL)   /* 160 */
#define POS_EMB_N   (SEQ_LEN * D_MODEL) /* 128 */
#define WQ_N        (D_MODEL * D_MODEL) /* 256 */
#define WK_N        (D_MODEL * D_MODEL) /* 256 */
#define WV_N        (D_MODEL * D_MODEL) /* 256 */
#define WOT_N       (D_MODEL * VOCAB)   /* 160 */

#define WORK_N      (3 * SEQ_LEN * D_MODEL + SEQ_LEN * SEQ_LEN)  /* 448 */

/* ===== Q16 weight storage (split hi/lo for 32-bit) ===== */
typedef struct {
    int16_t hi[WQ_N];
    int16_t lo[WQ_N];
} q16_pair;

/* All Q16 weight groups */
typedef struct {
    q16_pair tok_emb;   /* tok_emb hi/lo  (160) */
    q16_pair pos_emb;   /* pos_emb hi/lo  (128) */
    q16_pair wq;        /* Wq hi/lo       (256) */
    q16_pair wk;        /* Wk hi/lo       (256) */
    q16_pair wv;        /* Wv hi/lo       (256) */
    q16_pair wot;       /* Wout hi/lo     (160) */
} q16_weights;

/* All Q8 weight copies (for forward/backward) */
typedef struct {
    q8  tok_emb[TOK_EMB_N];
    q8  pos_emb[POS_EMB_N];
    q8  wq[WQ_N];
    q8  wk[WK_N];
    q8  wv[WV_N];
    q8  wot[WOT_N];
} q8_weights;

/* ===== Gradient accumulators (one Q15 word per weight) ===== */
/* PDP-11 up_do reads one word per weight; zerog clears N words per group. */
typedef struct {
    q15 dtke[TOK_EMB_N];
    q15 dpse[POS_EMB_N];
    q15 dwq[WQ_N];
    q15 dwk[WK_N];
    q15 dwv[WV_N];
    q15 dwot[WOT_N];
} gradients;

/* ===== Forward cache ===== */
typedef struct {
    q8  xx[SEQ_LEN * D_MODEL];   /* embeddings (input to attn) */
    q8  yy[SEQ_LEN * D_MODEL];   /* attn output + residual     */
    q8  logits[VOCAB * SEQ_LEN]; /* output logits              */
    q8  work[WORK_N];            /* attn workspace: Q|K|V|S    */
} forward_cache;

/* ===== Backward workspace ===== */
typedef struct {
    q15 dl[VOCAB];               /* dLogits (one position)     */
    q15 dy[SEQ_LEN * D_MODEL];   /* dY                         */
    q15 da[SEQ_LEN * SEQ_LEN];   /* dA / dSc                   */
    q15 dqq[SEQ_LEN * D_MODEL];  /* dQ                         */
    q15 dkk[SEQ_LEN * D_MODEL];  /* dK                         */
    q15 dvv[SEQ_LEN * D_MODEL];  /* dV                         */
    q15 dxx[SEQ_LEN * D_MODEL];  /* dX                         */
    q15 dtmp[SEQ_LEN];           /* temp vector (column extract) */
} backward_workspace;

/* ===== Training state ===== */
typedef struct {
    int   tr_stp;   /* current step */
    int   tr_hit;   /* correct predictions */
    int   tr_tot;   /* total predictions */
    int   tokens[SEQ_LEN];   /* input tokens */
    int   target[SEQ_LEN];   /* target (reversed) */
} training_state;

/* ===== Lookup tables ===== */
extern const q8   exptbl[256];   /* exp(-i/32) in Q8, i=0..255 */
extern const q12  logtbl[257];   /* -ln(x/256)*4096, x=0..256 */

/* ===== Global state ===== */
extern q16_weights    g_q16w;
extern q8_weights     g_q8w;
extern gradients      g_grad;
extern forward_cache  g_fwd;
extern backward_workspace g_bkw;
extern training_state g_trn;

/* ===== FXMATH.MAC — Level 0 scalar primitives ===== */
q8    fxdiv(q8 dividend, q8 divisor);
q16   fxmul(q8 a, q8 b);

/* ===== VECOP.MAC — Level 1 vector operations ===== */
q8    vdot(const q8 *x, const q8 *y, int len);
void  vadd(const q8 *x, const q8 *y, q8 *z, int len);
void  vsub(const q8 *x, const q8 *y, q8 *z, int len);
void  vscl(const q8 *x, q8 *y, int len, q8 alpha);
int   vmax(const q8 *vec, int len, int *out_idx);
void  vcpy(const q8 *src, q8 *dst, int len);
void  vclr(q8 *vec, int len);
void  vsadd(const q8 *src, q8 *dst, int len, q8 scalar);

/* ===== MATOP.MAC — Level 2 matrix-vector operations ===== */
void  mvmul(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols);
void  mvadd(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols);
void  vtmul(const q8 *mat, const q8 *vin, q8 *vout, int rows, int cols);
void  outer(const q8 *vx, const q8 *vy, q8 *mat, int rows, int cols);

/* ===== ACTFN.MAC — Activation functions ===== */
void  vrelu(q8 *vec, int len);
void  sftmx(q8 *vec, int len);

/* ===== LAYER.MAC — Neural network layers ===== */
void  embed(const int *tokens, const q8 *tkemb, const q8 *psemb,
            q8 *xout, int seq_len, int d_model);
void  proj(const q8 *yin, const q8 *wout, q8 *logits,
           int seq_len, int d_model, int vocab);
void  attn(const q8 *xin, const q8 *wq, const q8 *wk, const q8 *wv,
           q8 *yout, q8 *work, int seq_len, int d_model, int sqrtsh);

/* ===== FORWRD.MAC — Forward pass & weight conversion ===== */
void  cvt16(void);
void  forwrd(void);

/* ===== BKWRD.MAC — Backward pass ===== */
void  bkwrd(void);

/* ===== UPDAT.MAC — Weight init, SGD update, gradient zeroing ===== */
void  rand_seed(int seed);
int   rand_next(void);
void  initw(void);
void  updat(void);
void  zerog(void);

/* ===== TRAIN.MAC — Training loop ===== */
void  gensm(void);
void  count(void);
q12   closs(void);
void  report(void);
void  test(void);
int   train(void);

/* ===== I/O helpers ===== */
void  putc_char(char c);
void  puts_str(const char *s);
void  newln(void);
void  putoct(int val);
void  putdec(int val);
void  putq8(q8 val);
void  putspc(void);
void  putvec(const q8 *vec, int len);

#endif /* ATARI_AI_H */