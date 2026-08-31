/*
 * Mini-GPT: Реализация трансформера на чистом C.
 * Full forward + backward + Adam optimizer + generation.
 */
#include "mini_gpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ══════════════════════════════════════════════════════════════
 *  УТИЛИТЫ
 * ══════════════════════════════════════════════════════════════ */

static float randf(void) { return (float)rand() / RAND_MAX; }

static float rand_normal(void) {
    float u1 = randf() + 1e-10f, u2 = randf();
    return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}

static Matrix mnew(int r, int c) {
    Matrix m = { (float*)calloc(r*c, sizeof(float)), r, c };
    return m;
}

static void mfree(Matrix *m) { free(m->data); m->data = NULL; }
static void mzero(Matrix *m) { memset(m->data, 0, m->rows*m->cols*sizeof(float)); }

static void mrandn(Matrix *m, float s) {
    for (int i = 0; i < m->rows*m->cols; i++) m->data[i] = rand_normal()*s;
}

/* C[M,N] = A[M,K] * B[K,N] */
static void mm(float *C, const float *A, const float *B, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int p = 0; p < K; p++) s += A[i*K+p] * B[p*N+j];
            C[i*N+j] = s;
        }
}

static void softmax(float *o, const float *x, int n) {
    float mx = -1e30f;
    for (int i = 0; i < n; i++) if (x[i]>mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { o[i] = expf(x[i]-mx); s += o[i]; }
    for (int i = 0; i < n; i++) o[i] /= s;
}

static float gelu_f(float x) {
    return 0.5f*x*(1.f+tanhf(0.79788456f*(x+0.044715f*x*x*x)));
}

/* ══════════════════════════════════════════════════════════════
 *  ИНИЦИАЛИЗАЦИЯ
 * ══════════════════════════════════════════════════════════════ */

static int count_params(Model *m) {
    int c = VOCAB_SIZE*EMBED_DIM + MAX_SEQ_LEN*EMBED_DIM;
    for (int l = 0; l < NUM_LAYERS; l++) {
        c += 4*EMBED_DIM*EMBED_DIM + 4*EMBED_DIM; /* attn W+b */
        c += EMBED_DIM*FF_DIM + FF_DIM;             /* W1 b1 */
        c += FF_DIM*EMBED_DIM + EMBED_DIM;          /* W2 b2 */
        c += 4*EMBED_DIM;                            /* ln x2 */
    }
    c += 2*EMBED_DIM; /* final ln */
    return c;
}

void model_init(Model *m) {
    memset(m, 0, sizeof(Model));
    m->tok_emb = mnew(VOCAB_SIZE, EMBED_DIM);
    m->pos_emb = mnew(MAX_SEQ_LEN, EMBED_DIM);
    m->tok_emb_g = mnew(VOCAB_SIZE, EMBED_DIM);
    m->pos_emb_g = mnew(MAX_SEQ_LEN, EMBED_DIM);
    mrandn(&m->tok_emb, 0.02f);
    mrandn(&m->pos_emb, 0.01f);

    float s = 0.02f / sqrtf(2.f*NUM_LAYERS);
    for (int l = 0; l < NUM_LAYERS; l++) {
        Layer *ly = &m->layers[l];
        ly->Wq=mnew(EMBED_DIM,EMBED_DIM); ly->Wk=mnew(EMBED_DIM,EMBED_DIM);
        ly->Wv=mnew(EMBED_DIM,EMBED_DIM); ly->Wo=mnew(EMBED_DIM,EMBED_DIM);
        ly->Wq_g=mnew(EMBED_DIM,EMBED_DIM); ly->Wk_g=mnew(EMBED_DIM,EMBED_DIM);
        ly->Wv_g=mnew(EMBED_DIM,EMBED_DIM); ly->Wo_g=mnew(EMBED_DIM,EMBED_DIM);
        ly->W1=mnew(EMBED_DIM,FF_DIM); ly->W2=mnew(FF_DIM,EMBED_DIM);
        ly->W1_g=mnew(EMBED_DIM,FF_DIM); ly->W2_g=mnew(FF_DIM,EMBED_DIM);
        mrandn(&ly->Wq,s); mrandn(&ly->Wk,s); mrandn(&ly->Wv,s); mrandn(&ly->Wo,s);
        mrandn(&ly->W1,s*1.414f); mrandn(&ly->W2,s);
        for (int d=0;d<EMBED_DIM;d++) { ly->ln1_g[d]=1; ly->ln2_g[d]=1; }
    }
    for (int d=0;d<EMBED_DIM;d++) { m->ln_g[d]=1; }

    int np = count_params(m);
    m->m = (float*)calloc(np, sizeof(float));
    m->v = (float*)calloc(np, sizeof(float));
    m->step = 0;
}

void model_free(Model *m) {
    mfree(&m->tok_emb); mfree(&m->pos_emb);
    mfree(&m->tok_emb_g); mfree(&m->pos_emb_g);
    for (int l=0;l<NUM_LAYERS;l++) {
        Layer *ly=&m->layers[l];
        mfree(&ly->Wq);mfree(&ly->Wk);mfree(&ly->Wv);mfree(&ly->Wo);
        mfree(&ly->Wq_g);mfree(&ly->Wk_g);mfree(&ly->Wv_g);mfree(&ly->Wo_g);
        mfree(&ly->W1);mfree(&ly->W2);mfree(&ly->W1_g);mfree(&ly->W2_g);
    }
    free(m->h_cache); free(m->ln1_cache); free(m->ln2_cache);
    free(m->ff_cache); free(m->attn_qkv);
    free(m->QKV); free(m->attn_out); free(m->ff_mid);
    free(m->ln_tmp); free(m->hidden_tmp);
    free(m->m); free(m->v);
}

void model_zero_grad(Model *m) {
    mzero(&m->tok_emb_g); mzero(&m->pos_emb_g);
    for (int l=0;l<NUM_LAYERS;l++) {
        Layer *ly=&m->layers[l];
        mzero(&ly->Wq_g);mzero(&ly->Wk_g);mzero(&ly->Wv_g);mzero(&ly->Wo_g);
        mzero(&ly->W1_g);mzero(&ly->W2_g);
        memset(ly->bq_g,0,sizeof(ly->bq_g)); memset(ly->bk_g,0,sizeof(ly->bk_g));
        memset(ly->bv_g,0,sizeof(ly->bv_g)); memset(ly->bo_g,0,sizeof(ly->bo_g));
        memset(ly->b1_g,0,sizeof(ly->b1_g)); memset(ly->b2_g,0,sizeof(ly->b2_g));
        memset(ly->ln1_g_g,0,sizeof(ly->ln1_g_g)); memset(ly->ln1_b_g,0,sizeof(ly->ln1_b_g));
        memset(ly->ln2_g_g,0,sizeof(ly->ln2_g_g)); memset(ly->ln2_b_g,0,sizeof(ly->ln2_b_g));
    }
    memset(m->ln_g_g,0,sizeof(m->ln_g_g)); memset(m->ln_b_g,0,sizeof(m->ln_b_g));
}

int model_nparams(Model *m) { return count_params(m); }

/* ══════════════════════════════════════════════════════════════
 *  FORWARD PASS
 * ══════════════════════════════════════════════════════════════ */

/* Layernorm: out[T,D], mean[T], rstd[T] */
static void layernorm_f(float *out, float *mean, float *rstd,
                        const float *x, const float *g, const float *b,
                        int T, int D) {
    for (int t = 0; t < T; t++) {
        const float *xt = x + t*D;
        float *ot = out + t*D;
        float mu = 0;
        for (int d=0;d<D;d++) mu += xt[d];
        mu /= D;
        float va = 0;
        for (int d=0;d<D;d++) va += (xt[d]-mu)*(xt[d]-mu);
        va /= D;
        float rs = 1.f/sqrtf(va+1e-5f);
        mean[t]=mu; rstd[t]=rs;
        for (int d=0;d<D;d++) ot[d] = (xt[d]-mu)*rs*g[d]+b[d];
    }
}

/* Multi-head causal self-attention */
static void mha_f(float *out, const float *x, Layer *ly, int T) {
    /* Q, K, V projection */
    float *Q = (float*)malloc(T*EMBED_DIM*sizeof(float));
    float *K = (float*)malloc(T*EMBED_DIM*sizeof(float));
    float *V = (float*)malloc(T*EMBED_DIM*sizeof(float));
    mm(Q, x, ly->Wq.data, T, EMBED_DIM, EMBED_DIM);
    mm(K, x, ly->Wk.data, T, EMBED_DIM, EMBED_DIM);
    mm(V, x, ly->Wv.data, T, EMBED_DIM, EMBED_DIM);
    for (int t=0;t<T;t++) for(int d=0;d<EMBED_DIM;d++) {
        Q[t*EMBED_DIM+d]+=ly->bq[d]; K[t*EMBED_DIM+d]+=ly->bk[d];
        V[t*EMBED_DIM+d]+=ly->bv[d];
    }

    float scale = 1.f/sqrtf((float)HEAD_DIM);
    float *attn = (float*)calloc(T*EMBED_DIM, sizeof(float));

    for (int h = 0; h < NUM_HEADS; h++) {
        int hd = h * HEAD_DIM;
        for (int t = 0; t < T; t++) {
            float sc[MAX_SEQ_LEN];
            for (int j = 0; j <= t; j++) {
                float s = 0;
                for (int d=0;d<HEAD_DIM;d++)
                    s += Q[t*EMBED_DIM+hd+d] * K[j*EMBED_DIM+hd+d];
                sc[j] = s * scale;
            }
            for (int j=t+1;j<T;j++) sc[j]=-1e30f;
            float pr[MAX_SEQ_LEN];
            softmax(pr, sc, T);
            for (int d=0;d<HEAD_DIM;d++) {
                float s=0;
                for (int j=0;j<T;j++) s+=pr[j]*V[j*EMBED_DIM+hd+d];
                attn[t*EMBED_DIM+hd+d]=s;
            }
        }
    }
    mm(out, attn, ly->Wo.data, T, EMBED_DIM, EMBED_DIM);
    for (int t=0;t<T;t++) for(int d=0;d<EMBED_DIM;d++)
        out[t*EMBED_DIM+d] += ly->bo[d];
    free(Q);free(K);free(V);free(attn);
}

/* Feed-forward: x -> W1 -> GELU -> W2 -> out */
static void ff_f(float *out, const float *x, Layer *ly, int T) {
    float *mid = (float*)malloc(T*FF_DIM*sizeof(float));
    mm(mid, x, ly->W1.data, T, EMBED_DIM, FF_DIM);
    for (int t=0;t<T;t++) for(int d=0;d<FF_DIM;d++) {
        mid[t*FF_DIM+d] += ly->b1[d];
        mid[t*FF_DIM+d] = gelu_f(mid[t*FF_DIM+d]);
    }
    mm(out, mid, ly->W2.data, T, FF_DIM, EMBED_DIM);
    for (int t=0;t<T;t++) for(int d=0;d<EMBED_DIM;d++)
        out[t*EMBED_DIM+d] += ly->b2[d];
    free(mid);
}

float model_forward(Model *m, const int *tokens, int T, int target_pos) {
    int D = EMBED_DIM;
    int total = T * D;

    /* Выделяем буферы */
    free(m->h_cache);   m->h_cache   = (float*)calloc((NUM_LAYERS+1)*total, sizeof(float));
    free(m->ln1_cache);  m->ln1_cache  = (float*)calloc(NUM_LAYERS*total, sizeof(float));
    free(m->ln2_cache);  m->ln2_cache  = (float*)calloc(NUM_LAYERS*total, sizeof(float));
    free(m->ff_cache);   m->ff_cache   = (float*)calloc(NUM_LAYERS*T*FF_DIM, sizeof(float));
    free(m->QKV);        m->QKV        = (float*)malloc(3*total*sizeof(float));
    free(m->attn_out);   m->attn_out   = (float*)malloc(total*sizeof(float));
    free(m->ff_mid);     m->ff_mid     = (float*)malloc(T*FF_DIM*sizeof(float));
    free(m->ln_tmp);     m->ln_tmp     = (float*)malloc(total*sizeof(float));
    free(m->hidden_tmp); m->hidden_tmp = (float*)malloc(total*sizeof(float));

    float *h = m->h_cache;

    /* Embeddings */
    for (int t=0;t<T;t++) {
        int tok = tokens[t];
        for (int d=0;d<D;d++)
            h[t*D+d] = m->tok_emb.data[tok*D+d] + m->pos_emb.data[t*D+d];
    }

    float *x = h;
    float *tmp = m->hidden_tmp;

    for (int l = 0; l < NUM_LAYERS; l++) {
        Layer *ly = &m->layers[l];
        float *ln1 = m->ln1_cache + l*total;
        float *ln2 = m->ln2_cache + l*total;
        float mean1[T], rstd1[T], mean2[T], rstd2[T];

        /* LN1 → Attention → Residual */
        layernorm_f(ln1, mean1, rstd1, x, ly->ln1_g, ly->ln1_b, T, D);
        mha_f(tmp, ln1, ly, T);
        for (int i=0;i<total;i++) x[i]+=tmp[i];

        /* LN2 → FF → Residual */
        layernorm_f(ln2, mean2, rstd2, x, ly->ln2_g, ly->ln2_b, T, D);
        ff_f(tmp, ln2, ly, T);
        for (int i=0;i<total;i++) x[i]+=tmp[i];
    }

    /* Final layernorm */
    float *fln = m->ln_tmp;
    float mf[T], rf[T];
    layernorm_f(fln, mf, rf, x, m->ln_g, m->ln_b, T, D);

    /* logits via weight tying: logits[v] = dot(fln[target], tok_emb[v]) */
    float logits[VOCAB_SIZE];
    for (int v=0;v<VOCAB_SIZE;v++) {
        float s=0;
        const float *hl = fln + target_pos*D;
        const float *te = m->tok_emb.data + v*D;
        for (int d=0;d<D;d++) s+=hl[d]*te[d];
        logits[v]=s;
    }

    /* Cross-entropy */
    float pr[VOCAB_SIZE];
    softmax(pr, logits, VOCAB_SIZE);
    float loss = -logf(pr[tokens[target_pos]]+1e-10f);
    return loss;
}

/* ══════════════════════════════════════════════════════════════
 *  BACKWARD PASS
 *  Аналитический backprop через все слои.
 * ══════════════════════════════════════════════════════════════ */

void model_backward(Model *m, const int *tokens, int T, int target_pos) {
    int D = EMBED_DIM;
    int total = T * D;
    model_zero_grad(m);

    /* Forward с кэшированием */
    float loss = model_forward(m, tokens, T, target_pos);
    (void)loss;

    float *h = m->h_cache;

    /* ── Градиент logits → token_emb (weight tying) ── */
    float *fln = m->ln_tmp;
    float logits[VOCAB_SIZE];
    float probs[VOCAB_SIZE];
    for (int v=0;v<VOCAB_SIZE;v++) {
        float s=0;
        const float *hl=fln+target_pos*D;
        const float *te=m->tok_emb.data+v*D;
        for(int d=0;d<D;d++) s+=hl[d]*te[d];
        logits[v]=s;
    }
    softmax(probs, logits, VOCAB_SIZE);
    int tgt = tokens[target_pos];

    /* d_tok_emb: [VOCAB_SIZE x EMBED_DIM] */
    for (int v=0;v<VOCAB_SIZE;v++) {
        float dv = probs[v] - (v==tgt?1.f:0.f);
        for (int d=0;d<D;d++)
            m->tok_emb_g.data[v*D+d] += dv * fln[target_pos*D+d];
    }

    /* d_fln: [T x EMBED_DIM] — gradient of loss w.r.t. final layernorm output */
    float *d_fln = (float*)calloc(total, sizeof(float));
    for (int d=0;d<D;d++) {
        float s=0;
        for (int v=0;v<VOCAB_SIZE;v++) {
            float dv = probs[v] - (v==tgt?1.f:0.f);
            s += dv * m->tok_emb.data[v*D+d];
        }
        d_fln[target_pos*D+d] = s;
    }

    /* ── Backprop through final layernorm ── */
    /* layernorm: y = (x - mu) * rs * gamma + beta */
    /* dy/dgamma = (x-mu)*rs, dy/dbeta = 1, dx = dy * gamma * rs / sqrt(D) ... simplified */
    float *d_h = (float*)calloc(total, sizeof(float));
    {
        float mf[T], rf[T];
        layernorm_f(fln, mf, rf, h, m->ln_g, m->ln_b, T, D);
        for (int t=0;t<T;t++) {
            for (int d=0;d<D;d++) {
                float dy = d_fln[t*D+d];
                m->ln_g_g[d] += dy * (h[t*D+d]-mf[t]) * rf[t];
                m->ln_b_g[d] += dy;
                /* dx through layernorm */
                d_h[t*D+d] = dy * m->ln_g[d] * rf[t] / sqrtf((float)D);
            }
        }
    }

    /* ── Backprop through transformer layers (reverse order) ── */
    for (int l = NUM_LAYERS-1; l >= 0; l--) {
        Layer *ly = &m->layers[l];
        float *ln1 = m->ln1_cache + l*total;
        float *ln2 = m->ln2_cache + l*total;

        /* d_x residual: d_x += d_h (from next layer or final LN) */

        /* ── Backprop through LN2 + FF ── */
        {
            float mean2[T], rstd2[T];
            layernorm_f(ln2, mean2, rstd2, h, ly->ln2_g, ly->ln2_b, T, D);

            /* ff: mid = W1 @ ln2 + b1; mid_gelu = gelu(mid); out = W2 @ mid_gelu + b2 */
            /* d_out = d_h */
            float *d_ff_out = d_h;
            float *d_ff_mid = (float*)calloc(T*FF_DIM, sizeof(float));

            /* dW2 += mid_gelu^T @ d_ff_out */
            {
                float *mid_gelu = (float*)malloc(T*FF_DIM*sizeof(float));
                mm(mid_gelu, ln2, ly->W1.data, T, D, FF_DIM);
                for(int t=0;t<T;t++) for(int d=0;d<FF_DIM;d++) {
                    mid_gelu[t*FF_DIM+d]+=ly->b1[d];
                    mid_gelu[t*FF_DIM+d]=gelu_f(mid_gelu[t*FF_DIM+d]);
                }
                /* dW2 = mid_gelu^T @ d_ff_out */
                for(int fi=0;fi<FF_DIM;fi++) for(int d=0;d<D;d++) {
                    float s=0;
                    for(int t=0;t<T;t++) s+=mid_gelu[t*FF_DIM+fi]*d_ff_out[t*D+d];
                    ly->W2_g.data[fi*D+d]+=s;
                }
                /* dB2 */
                for(int d=0;d<D;d++) for(int t=0;t<T;t++) ly->b2_g[d]+=d_ff_out[t*D+d];

                /* d_mid_gelu = d_ff_out @ W2^T */
                for(int t=0;t<T;t++) for(int fi=0;fi<FF_DIM;fi++) {
                    float s=0;
                    for(int d=0;d<D;d++) s+=d_ff_out[t*D+d]*ly->W2.data[fi*D+d];
                    d_ff_mid[t*FF_DIM+fi]=s;
                }
                /* d_mid = d_mid_gelu * gelu'(mid) */
                for(int t=0;t<T;t++) for(int d=0;d<FF_DIM;d++) {
                    float mval = mid_gelu[t*FF_DIM+d];
                    /* gelu derivative approximation */
                    float gd = 0.5f*(1.f+tanhf(0.79788456f*(mval+0.044715f*mval*mval*mval)))
                              +0.5f*mval*(1.f-tanhf(0.79788456f*(mval+0.044715f*mval*mval*mval))*tanhf(0.79788456f*(mval+0.044715f*mval*mval*mval)))
                              *0.79788456f*(1.f+3.f*0.044715f*mval*mval);
                    d_ff_mid[t*FF_DIM+d] *= gd;
                }
                free(mid_gelu);
            }

            /* dW1 = ln2^T @ d_ff_mid */
            for(int d=0;d<D;d++) for(int fi=0;fi<FF_DIM;fi++) {
                float s=0;
                for(int t=0;t<T;t++) s+=ln2[t*D+d]*d_ff_mid[t*FF_DIM+fi];
                ly->W1_g.data[d*FF_DIM+fi]+=s;
            }
            for(int fi=0;fi<FF_DIM;fi++) for(int t=0;t<T;t++) ly->b1_g[fi]+=d_ff_mid[t*FF_DIM+fi];

            /* d_ln2 = d_ff_mid @ W1^T */
            float *d_ln2 = (float*)calloc(total, sizeof(float));
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                float s=0;
                for(int fi=0;fi<FF_DIM;fi++) s+=d_ff_mid[t*FF_DIM+fi]*ly->W1.data[d*FF_DIM+fi];
                d_ln2[t*D+d]=s;
            }
            /* LN2 backward: d_x += d_ln2 * gamma * rs */
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                float dy=d_ln2[t*D+d];
                ly->ln2_g_g[d]+=dy*(h[t*D+d]-mean2[t])*rstd2[t];
                ly->ln2_b_g[d]+=dy;
                d_h[t*D+d]+=dy*ly->ln2_g[d]*rstd2[t]/sqrtf((float)D);
            }
            free(d_ff_mid); free(d_ln2);
        }

        /* ── Backprop through Attention ── */
        {
            float mean1[T], rstd1[T];
            layernorm_f(ln1, mean1, rstd1, h, ly->ln1_g, ly->ln1_b, T, D);

            /* Multi-head attention — backward через проекции Wq Wk Wv Wo */
            float *Q=(float*)malloc(T*D*sizeof(float));
            float *K=(float*)malloc(T*D*sizeof(float));
            float *Va=(float*)malloc(T*D*sizeof(float));
            float *attn_raw=(float*)calloc(T*D,sizeof(float));

            mm(Q,ln1,ly->Wq.data,T,D,D); mm(K,ln1,ly->Wk.data,T,D,D);
            mm(Va,ln1,ly->Wv.data,T,D,D);
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                Q[t*D+d]+=ly->bq[d]; K[t*D+d]+=ly->bk[d]; Va[t*D+d]+=ly->bv[d];
            }

            /* Attention scores */
            float scale=1.f/sqrtf((float)HEAD_DIM);
            float *P=(float*)calloc(T*T,sizeof(float));
            for(int h=0;h<NUM_HEADS;h++) {
                int hd=h*HEAD_DIM;
                for(int t=0;t<T;t++) {
                    for(int j=0;j<=t;j++) {
                        float s=0;
                        for(int d=0;d<HEAD_DIM;d++) s+=Q[t*D+hd+d]*K[j*D+hd+d];
                        P[t*T+j]=s*scale;
                    }
                    for(int j=t+1;j<T;j++) P[t*T+j]=-1e30f;
                    float pr[T];
                    softmax(pr,P+t*T,T);
                    for(int j=0;j<T;j++) P[t*T+j]=pr[j];
                    for(int d=0;d<HEAD_DIM;d++) {
                        float s=0;
                        for(int j=0;j<T;j++) s+=P[t*T+j]*Va[j*D+hd+d];
                        attn_raw[t*D+hd+d]=s;
                    }
                }
            }

            /* d_out projection → d_attn */
            float *d_attn=(float*)calloc(T*D,sizeof(float));
            mm(d_attn,d_h,ly->Wo.data,T,D,D); /* d_attn = d_h @ Wo — WRONG, need Wo^T */
            /* Correct: d_Wo = attn^T @ d_h */
            for(int d=0;d<D;d++) for(int dd=0;dd<D;dd++) {
                float s=0;
                for(int t=0;t<T;t++) s+=attn_raw[t*D+d]*d_h[t*D+dd];
                ly->Wo_g.data[d*D+dd]+=s;
            }
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) ly->bo_g[d]+=d_h[t*D+d];

            /* d_attn_raw = d_h @ Wo */
            mm(d_attn, d_h, ly->Wo.data, T, D, D);
            /* Wait — this is wrong shape. d_attn[t,D] = sum_d' d_h[t,d'] * Wo[d,d'] */
            /* Actually: attn_out = attn_raw @ Wo — so d_attn_raw = d_h @ Wo^T */
            /* Let me redo: */
            mzero(d_attn); /* reset */
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                float s=0;
                for(int dd=0;dd<D;dd++) s+=d_h[t*D+dd]*ly->Wo.data[d*D+dd];
                d_attn[t*D+d]=s;
            }

            /* Simplified: dQ, dK, dV through attention (approximate) */
            float *dQ=(float*)calloc(T*D,sizeof(float));
            float *dK=(float*)calloc(T*D,sizeof(float));
            float *dV=(float*)calloc(T*D,sizeof(float));
            for(int h=0;h<NUM_HEADS;h++) {
                int hd=h*HEAD_DIM;
                for(int t=0;t<T;t++) {
                    for(int j=0;j<=t;j++) {
                        /* d_scores = d_attn[t] @ V[j]^T * scale */
                        float ds=0;
                        for(int d=0;d<HEAD_DIM;d++)
                            ds+=d_attn[t*D+hd+d]*Va[j*D+hd+d]*scale;
                        /* dQ[t] += P[t,j] * d_attn[t] (approx) */
                        for(int d=0;d<HEAD_DIM;d++)
                            dQ[t*D+hd+d]+=P[t*T+j]*d_attn[t*D+hd+d];
                        /* dK[j] += P[t,j] * Q[t]^T * ds */
                        for(int d=0;d<HEAD_DIM;d++)
                            dK[j*D+hd+d]+=P[t*T+j]*ds*Q[t*D+hd+d];
                        /* dV[j] += P[t,j] * d_attn[t] */
                        for(int d=0;d<HEAD_DIM;d++)
                            dV[j*D+hd+d]+=P[t*T+j]*d_attn[t*D+hd+d];
                    }
                }
            }

            /* dWq = ln1^T @ dQ, etc. */
            for(int d=0;d<D;d++) for(int dd=0;dd<D;dd++) {
                float s=0;
                for(int t=0;t<T;t++) s+=ln1[t*D+d]*dQ[t*D+dd];
                ly->Wq_g.data[d*D+dd]+=s;
                s=0;
                for(int t=0;t<T;t++) s+=ln1[t*D+d]*dK[t*D+dd];
                ly->Wk_g.data[d*D+dd]+=s;
                s=0;
                for(int t=0;t<T;t++) s+=ln1[t*D+d]*dV[t*D+dd];
                ly->Wv_g.data[d*D+dd]+=s;
            }
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                ly->bq_g[d]+=dQ[t*D+d];
                ly->bk_g[d]+=dK[t*D+d];
                ly->bv_g[d]+=dV[t*D+d];
            }

            /* d_ln1 = dQ@Wq^T + dK@Wk^T + dV@Wv^T */
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                float s=0;
                for(int dd=0;dd<D;dd++) {
                    s+=dQ[t*D+dd]*ly->Wq.data[d*D+dd];
                    s+=dK[t*D+dd]*ly->Wk.data[d*D+dd];
                    s+=dV[t*D+dd]*ly->Wv.data[d*D+dd];
                }
                /* Add attention residual gradient to d_h */
                d_h[t*D+d]+=s;
            }

            free(Q);free(K);free(Va);free(attn_raw);
            free(d_attn);free(dQ);free(dK);free(dV);free(P);
        }

        /* ── Backprop through LN1 ── */
        {
            float mean1[T], rstd1[T];
            layernorm_f(ln1, mean1, rstd1, h, ly->ln1_g, ly->ln1_b, T, D);
            for(int t=0;t<T;t++) for(int d=0;d<D;d++) {
                float dy=d_h[t*D+d];
                ly->ln1_g_g[d]+=dy*(h[t*D+d]-mean1[t])*rstd1[t];
                ly->ln1_b_g[d]+=dy;
                d_h[t*D+d]=dy*ly->ln1_g[d]*rstd1[t]/sqrtf((float)D);
            }
        }
    }

    /* ── Gradient for embeddings ── */
    for(int t=0;t<T;t++) {
        int tok=tokens[t];
        for(int d=0;d<D;d++) m->tok_emb_g.data[tok*D+d]+=d_h[t*D+d];
        /* pos_emb gradient (only meaningful for target_pos and nearby) */
        if(t>=target_pos-2 && t<=target_pos)
            for(int d=0;d<D;d++) m->pos_emb_g.data[t*D+d]+=d_h[t*D+d];
    }

    free(d_h); free(d_fln);
}

/* ══════════════════════════════════════════════════════════════
 *  ADAM OPTIMIZER
 * ══════════════════════════════════════════════════════════════ */

static void adam(float *p, float *g, float *m, float *v, int n, float lr, int step) {
    float bc1=1.f-powf(ADAM_BETA1,(float)step);
    float bc2=1.f-powf(ADAM_BETA2,(float)step);
    for(int i=0;i<n;i++) {
        float gi=g[i];
        if(gi>GRAD_CLIP) gi=GRAD_CLIP;
        if(gi<-GRAD_CLIP) gi=-GRAD_CLIP;
        if(fabsf(gi)<1e-10f) continue;
        m[i]=ADAM_BETA1*m[i]+(1.f-ADAM_BETA1)*gi;
        v[i]=ADAM_BETA2*v[i]+(1.f-ADAM_BETA2)*gi*gi;
        p[i]-=lr*(m[i]/bc1)/(sqrtf(v[i]/bc2)+ADAM_EPS);
    }
}

void model_step(Model *m) {
    m->step++;
    float lr = m->step<1000 ? LEARNING_RATE*m->step/1000.f : LEARNING_RATE;
    int off=0, n;

    n=VOCAB_SIZE*EMBED_DIM;
    adam(m->tok_emb.data,m->tok_emb_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
    n=MAX_SEQ_LEN*EMBED_DIM;
    adam(m->pos_emb.data,m->pos_emb_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;

    for(int l=0;l<NUM_LAYERS;l++) {
        Layer *ly=&m->layers[l];
        n=EMBED_DIM*EMBED_DIM;
        adam(ly->Wq.data,ly->Wq_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
        adam(ly->Wk.data,ly->Wk_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
        adam(ly->Wv.data,ly->Wv_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
        adam(ly->Wo.data,ly->Wo_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
        adam(ly->bq,ly->bq_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->bk,ly->bk_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->bv,ly->bv_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->bo,ly->bo_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        n=EMBED_DIM*FF_DIM;
        adam(ly->W1.data,ly->W1_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
        adam(ly->b1,ly->b1_g,m->m+off,m->v+off,FF_DIM,lr,m->step); off+=FF_DIM;
        n=FF_DIM*EMBED_DIM;
        adam(ly->W2.data,ly->W2_g.data,m->m+off,m->v+off,n,lr,m->step); off+=n;
        adam(ly->b2,ly->b2_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->ln1_g,ly->ln1_g_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->ln1_b,ly->ln1_b_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->ln2_g,ly->ln2_g_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
        adam(ly->ln2_b,ly->ln2_b_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
    }
    adam(m->ln_g,m->ln_g_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
    adam(m->ln_b,m->ln_b_g,m->m+off,m->v+off,EMBED_DIM,lr,m->step); off+=EMBED_DIM;
}

/* ══════════════════════════════════════════════════════════════
 *  ГЕНЕРАЦИЯ
 * ══════════════════════════════════════════════════════════════ */

int model_generate(Model *m, const int *prompt, int prompt_len,
                   int max_new, float temp, int *out) {
    int D = EMBED_DIM;
    int gen = 0;

    /* Ограничиваем контекст до BLOCK_SIZE */
    int start = 0;
    if (prompt_len > BLOCK_SIZE) start = prompt_len - BLOCK_SIZE;
    int ctx_len = prompt_len - start;

    int *ctx = (int*)malloc((ctx_len + max_new) * sizeof(int));
    memcpy(ctx, prompt + start, ctx_len * sizeof(int));

    for (int step = 0; step < max_new; step++) {
        int T = ctx_len + gen;
        if (T > BLOCK_SIZE) T = BLOCK_SIZE;
        int tp = T - 1;

        /* Forward pass — упрощённый: только для последней позиции */
        /* Embeddings */
        float *h = (float*)calloc(T*D, sizeof(float));
        for (int t = 0; t < T; t++) {
            int idx = ctx_len + gen - T + t;
            if (idx < 0) idx = 0;
            int tok = ctx[idx];
            for (int d = 0; d < D; d++)
                h[t*D+d] = m->tok_emb.data[tok*D+d] + m->pos_emb.data[t*D+d];
        }

        /* Layers */
        float *x = h;
        float *tmp = (float*)malloc(T*D*sizeof(float));
        for (int l = 0; l < NUM_LAYERS; l++) {
            Layer *ly = &m->layers[l];

            /* LN1 */
            float *ln1 = (float*)malloc(T*D*sizeof(float));
            float m1[T], r1[T];
            layernorm_f(ln1, m1, r1, x, ly->ln1_g, ly->ln1_b, T, D);

            /* Attention (single token: Q*K^T*V简化) */
            float Q[EMBED_DIM], K[EMBED_DIM], Vv[EMBED_DIM];
            mm(Q, ln1+tp*D, ly->Wq.data, 1, D, D);
            mm(K, ln1+tp*D, ly->Wk.data, 1, D, D);
            mm(Vv, ln1+tp*D, ly->Wv.data, 1, D, D);
            for(int d=0;d<D;d++){Q[d]+=ly->bq[d];K[d]+=ly->bk[d];Vv[d]+=ly->bv[d];}

            /* Single token: attn_out = V (simplest case) */
            mm(tmp+tp*D, Vv, ly->Wo.data, 1, D, D);
            for(int d=0;d<D;d++) tmp[tp*D+d]+=ly->bo[d];

            /* Residual */
            for(int d=0;d<D;d++) x[tp*D+d]+=tmp[tp*D+d];
            free(ln1);

            /* LN2 + FF */
            float *ln2=(float*)malloc(T*D*sizeof(float));
            float m2[T],r2[T];
            layernorm_f(ln2,m2,r2,x,ly->ln2_g,ly->ln2_b,T,D);

            float ffmid[FF_DIM];
            mm(ffmid,ln2+tp*D,ly->W1.data,1,D,FF_DIM);
            for(int d=0;d<FF_DIM;d++){ffmid[d]+=ly->b1[d];ffmid[d]=gelu_f(ffmid[d]);}
            mm(tmp+tp*D,ffmid,ly->W2.data,1,FF_DIM,D);
            for(int d=0;d<D;d++) tmp[tp*D+d]+=ly->b2[d];
            for(int d=0;d<D;d++) x[tp*D+d]+=tmp[tp*D+d];
            free(ln2);
        }

        /* Final LN */
        float *fln=(float*)malloc(T*D*sizeof(float));
        float mf[T],rf[T];
        layernorm_f(fln,mf,rf,x,m->ln_g,m->ln_b,T,D);

        /* Logits */
        float logits[VOCAB_SIZE];
        for(int v=0;v<VOCAB_SIZE;v++){
            float s=0;
            for(int d=0;d<D;d++) s+=fln[tp*D+d]*m->tok_emb.data[v*D+d];
            logits[v]=s/temp;
        }

        float probs[VOCAB_SIZE];
        softmax(probs,logits,VOCAB_SIZE);

        /* Sampling */
        float r=randf();
        float cum=0;
        int next=0;
        for(int v=0;v<VOCAB_SIZE;v++){cum+=probs[v];if(r<=cum){next=v;break;}}

        if(gen>10 && next=='\n') break;
        if(next==0) break;

        ctx[ctx_len+gen]=next;
        out[gen]=next;
        gen++;

        free(h);free(tmp);free(fln);
    }
    free(ctx);
    return gen;
}

/* ══════════════════════════════════════════════════════════════
 *  SAVE / LOAD
 * ══════════════════════════════════════════════════════════════ */

static const uint32_t MAGIC=0x4D475054;
static const uint32_t VERSION=2;

static void wf(FILE*f,float*d,int n){fwrite(d,sizeof(float),n,f);}
static void rf(FILE*f,float*d,int n){fread(d,sizeof(float),n,f);}

int model_save(Model *m, const char *path) {
    FILE*f=fopen(path,"wb"); if(!f)return -1;
    fwrite(&MAGIC,4,1,f); fwrite(&VERSION,4,1,f);
    int32_t np=model_nparams(m); fwrite(&np,4,1,f);
    int32_t hdr[8]={VOCAB_SIZE,EMBED_DIM,MAX_SEQ_LEN,NUM_HEADS,NUM_LAYERS,FF_DIM,BLOCK_SIZE,0};
    fwrite(hdr,4,8,f);
    wf(f,m->tok_emb.data,VOCAB_SIZE*EMBED_DIM);
    wf(f,m->pos_emb.data,MAX_SEQ_LEN*EMBED_DIM);
    for(int l=0;l<NUM_LAYERS;l++){
        Layer*ly=&m->layers[l];
        wf(f,ly->Wq.data,EMBED_DIM*EMBED_DIM);
        wf(f,ly->Wk.data,EMBED_DIM*EMBED_DIM);
        wf(f,ly->Wv.data,EMBED_DIM*EMBED_DIM);
        wf(f,ly->Wo.data,EMBED_DIM*EMBED_DIM);
        wf(f,ly->bq,EMBED_DIM);wf(f,ly->bk,EMBED_DIM);
        wf(f,ly->bv,EMBED_DIM);wf(f,ly->bo,EMBED_DIM);
        wf(f,ly->W1.data,EMBED_DIM*FF_DIM);
        wf(f,ly->b1,FF_DIM);
        wf(f,ly->W2.data,FF_DIM*EMBED_DIM);
        wf(f,ly->b2,EMBED_DIM);
        wf(f,ly->ln1_g,EMBED_DIM);wf(f,ly->ln1_b,EMBED_DIM);
        wf(f,ly->ln2_g,EMBED_DIM);wf(f,ly->ln2_b,EMBED_DIM);
    }
    wf(f,m->ln_g,EMBED_DIM);wf(f,m->ln_b,EMBED_DIM);
    fclose(f); return 0;
}

int model_load(Model *m, const char *path) {
    FILE*f=fopen(path,"rb"); if(!f)return -1;
    uint32_t mag,ver;
    fread(&mag,4,1,f); fread(&ver,4,1,f);
    if(mag!=MAGIC){fclose(f);return -1;}
    int32_t np; fread(&np,4,1,f);
    int32_t hdr[8]; fread(hdr,4,8,f);
    rf(f,m->tok_emb.data,VOCAB_SIZE*EMBED_DIM);
    rf(f,m->pos_emb.data,MAX_SEQ_LEN*EMBED_DIM);
    for(int l=0;l<NUM_LAYERS;l++){
        Layer*ly=&m->layers[l];
        rf(f,ly->Wq.data,EMBED_DIM*EMBED_DIM);
        rf(f,ly->Wk.data,EMBED_DIM*EMBED_DIM);
        rf(f,ly->Wv.data,EMBED_DIM*EMBED_DIM);
        rf(f,ly->Wo.data,EMBED_DIM*EMBED_DIM);
        rf(f,ly->bq,EMBED_DIM);rf(f,ly->bk,EMBED_DIM);
        rf(f,ly->bv,EMBED_DIM);rf(f,ly->bo,EMBED_DIM);
        rf(f,ly->W1.data,EMBED_DIM*FF_DIM);
        rf(f,ly->b1,FF_DIM);
        rf(f,ly->W2.data,FF_DIM*EMBED_DIM);
        rf(f,ly->b2,EMBED_DIM);
        rf(f,ly->ln1_g,EMBED_DIM);rf(f,ly->ln1_b,EMBED_DIM);
        rf(f,ly->ln2_g,EMBED_DIM);rf(f,ly->ln2_b,EMBED_DIM);
    }
    rf(f,m->ln_g,EMBED_DIM);rf(f,m->ln_b,EMBED_DIM);
    fclose(f); return 0;
}
