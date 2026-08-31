/*
 * Mini-GPT: Языковая модель трансформер на чистом C.
 * GPT-2 архитектура, символьная токенизация, Adam оптимизатор.
 * Нулевые внешние зависимости — только stdlib + math.
 */
#ifndef MINI_GPT_H
#define MINI_GPT_H

#include <stdint.h>

/* ── Конфигурация ─────────────────────────────────────────────── */
#define VOCAB_SIZE    256
#define EMBED_DIM     96
#define MAX_SEQ_LEN   512
#define NUM_HEADS     4
#define NUM_LAYERS    4
#define FF_DIM        384
#define BLOCK_SIZE    128
#define LEARNING_RATE 0.0004f
#define ADAM_BETA1    0.9f
#define ADAM_BETA2    0.999f
#define ADAM_EPS      1e-8f
#define GRAD_CLIP     1.0f
#define HEAD_DIM      (EMBED_DIM / NUM_HEADS)

/* ── Матрица ──────────────────────────────────────────────────── */
typedef struct {
    float *data;
    int rows, cols;
} Matrix;

/* ── Один слой трансформера ──────────────────────────────────── */
typedef struct {
    /* Attention: Q K V O */
    Matrix Wq, Wk, Wv, Wo;
    float bq[EMBED_DIM], bk[EMBED_DIM], bv[EMBED_DIM], bo[EMBED_DIM];
    Matrix Wq_g, Wk_g, Wv_g, Wo_g;
    float bq_g[EMBED_DIM], bk_g[EMBED_DIM], bv_g[EMBED_DIM], bo_g[EMBED_DIM];

    /* Feed-forward */
    Matrix W1, W2;
    float b1[FF_DIM], b2[EMBED_DIM];
    Matrix W1_g, W2_g;
    float b1_g[FF_DIM], b2_g[EMBED_DIM];

    /* LayerNorm x2 */
    float ln1_g[EMBED_DIM], ln1_b[EMBED_DIM];
    float ln2_g[EMBED_DIM], ln2_b[EMBED_DIM];
    float ln1_g_g[EMBED_DIM], ln1_b_g[EMBED_DIM];
    float ln2_g_g[EMBED_DIM], ln2_b_g[EMBED_DIM];
} Layer;

/* ── Модель ──────────────────────────────────────────────────── */
typedef struct {
    Matrix tok_emb, pos_emb;
    Matrix tok_emb_g, pos_emb_g;
    Layer layers[NUM_LAYERS];
    float ln_g[EMBED_DIM], ln_b[EMBED_DIM];
    float ln_g_g[EMBED_DIM], ln_b_g[EMBED_DIM];

    /* Кэши для backward */
    float *h_cache;     /* [NUM_LAYERS+1][BLOCK_SIZE*EMBED_DIM] */
    float *ln1_cache;   /* [NUM_LAYERS][BLOCK_SIZE*EMBED_DIM] */
    float *ln2_cache;   /* [NUM_LAYERS][BLOCK_SIZE*EMBED_DIM] */
    float *ff_cache;    /* [NUM_LAYERS][BLOCK_SIZE*FF_DIM] */
    float *attn_qkv;    /* [NUM_LAYERS][3*BLOCK_SIZE*EMBED_DIM] */

    /* Adam */
    float *m, *v;
    int step;

    /* Вспомогательные буферы (выделяются перед forward) */
    int buf_T;
    float *QKV;         /* [T * 3 * EMBED_DIM] — Q K V все головы */
    float *attn_out;    /* [T * EMBED_DIM] */
    float *ff_mid;      /* [T * FF_DIM] */
    float *ln_tmp;      /* [T * EMBED_DIM] */
    float *hidden_tmp;  /* [T * EMBED_DIM] */
} Model;

/* ── API ──────────────────────────────────────────────────────── */
void model_init(Model *m);
void model_free(Model *m);
void model_zero_grad(Model *m);
int  model_nparams(Model *m);

/* Forward: возвращает loss для target позиции */
float model_forward(Model *m, const int *tokens, int T, int target_pos);

/* Backward: заполняет градиенты */
void model_backward(Model *m, const int *tokens, int T, int target_pos);

/* Adam update */
void model_step(Model *m);

/* Autoregressive генерация */
int  model_generate(Model *m, const int *prompt, int prompt_len,
                    int max_new_tokens, float temp, int *out);

/* Save / Load */
int model_save(Model *m, const char *path);
int model_load(Model *m, const char *path);

#endif
