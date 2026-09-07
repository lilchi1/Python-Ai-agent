/*
 * Mini-GPT: Языковая модель трансформер на чистом C.
 * GPT-2 архитектура, символьная токенизация, Adam оптимизатор.
 * Нулевые внешние зависимости — только stdlib + math.
 */
#ifndef MINI_GPT_H
#define MINI_GPT_H

#include <stdint.h>

#define VOCAB_SIZE    128
#define EMBED_DIM     96
#define MAX_SEQ_LEN   256
#define NUM_HEADS     4
#define NUM_LAYERS    4
#define FF_DIM        384
#define BLOCK_SIZE    128
#define LEARNING_RATE 0.0003f
#define ADAM_BETA1    0.9f
#define ADAM_BETA2    0.999f
#define ADAM_EPS      1e-8f
#define GRAD_CLIP     1.0f
#define HEAD_DIM      (EMBED_DIM / NUM_HEADS)
#define WARMUP_STEPS  100
#define MAX_TRAIN_STEPS 40000
#define LR_MIN_FRAC   0.05f

typedef struct {
    float *data;
    int rows, cols;
} Matrix;

typedef struct {
    Matrix Wq, Wk, Wv, Wo;
    float bq[EMBED_DIM], bk[EMBED_DIM], bv[EMBED_DIM], bo[EMBED_DIM];
    Matrix Wq_g, Wk_g, Wv_g, Wo_g;
    float bq_g[EMBED_DIM], bk_g[EMBED_DIM], bv_g[EMBED_DIM], bo_g[EMBED_DIM];

    Matrix W1, W2;
    float b1[FF_DIM], b2[EMBED_DIM];
    Matrix W1_g, W2_g;
    float b1_g[FF_DIM], b2_g[EMBED_DIM];

    float ln1_g[EMBED_DIM], ln1_b[EMBED_DIM];
    float ln2_g[EMBED_DIM], ln2_b[EMBED_DIM];
    float ln1_g_g[EMBED_DIM], ln1_b_g[EMBED_DIM];
    float ln2_g_g[EMBED_DIM], ln2_b_g[EMBED_DIM];
} Layer;

typedef struct {
    Matrix tok_emb, pos_emb;
    Matrix tok_emb_g, pos_emb_g;
    Layer layers[NUM_LAYERS];
    float ln_g[EMBED_DIM], ln_b[EMBED_DIM];
    float ln_g_g[EMBED_DIM], ln_b_g[EMBED_DIM];

    float *h_cache;
    float *ln1_cache;
    float *ln2_cache;
    float *ff_cache;
    float *attn_qkv;

    float *m, *v;
    int step;
    int total_params;
    float last_loss;

    int buf_T;
    float *QKV;
    float *attn_out;
    float *ff_mid;
    float *ln_tmp;
    float *hidden_tmp;
} Model;

void model_init(Model *m);
void model_free(Model *m);
void model_zero_grad(Model *m);
int  model_nparams(Model *m);

float model_forward(Model *m, const int *tokens, int T, int target_pos);
void model_backward(Model *m, const int *tokens, int T, int target_pos);
void model_step(Model *m);

int  model_generate(Model *m, const int *prompt, int prompt_len,
                    int max_new_tokens, float temp, int *out);

int model_save(Model *m, const char *path);
int model_load(Model *m, const char *path);

/* Training API */
typedef struct {
    float learning_rate;
    int   batch_size;
    int   epochs;
    int   max_seq_len;
    int   print_every;
    int   save_every;
    const char *save_path;
} TrainConfig;

TrainConfig train_config_default(void);
float train_step(Model *m, const int *tokens, int T, TrainConfig *cfg);
void train_on_data(Model *m, const int *data, int data_len, TrainConfig *cfg);

/* Token data loading */
int *load_text_file(const char *path, int *out_len);
int *load_directory(const char *dir, int *out_len);

/* Batch generation helpers */
void sample_batch(const int *data, int data_len, int B, int T,
                  int *out_x, int *out_y);

#endif
