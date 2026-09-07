/*
 * Mini-GPT Host: Обёртка для вызова модели из других языков (ctypes/Python).
 * Выделяет Model в куче, экспортирует чистые функции без struct в сигнатуре.
 */
#include "mini_gpt.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Инициализирует новую модель, возвращает указатель (void*) */
void *host_init(void) {
    Model *m = (Model*)malloc(sizeof(Model));
    if (!m) return NULL;
    model_init(m);
    return (void*)m;
}

void host_free(void *h) {
    if (!h) return;
    Model *m = (Model*)h;
    model_free(m);
    free(m);
}

int host_load(void *h, const char *path) {
    if (!h) return -1;
    return model_load((Model*)h, path);
}

int host_save(void *h, const char *path) {
    if (!h) return -1;
    return model_save((Model*)h, path);
}

int host_nparams(void *h) {
    if (!h) return 0;
    return model_nparams((Model*)h);
}

int host_generate(void *h, const int *prompt, int prompt_len,
                  int max_new, float temp, int *out) {
    if (!h) return 0;
    return model_generate((Model*)h, prompt, prompt_len, max_new, temp, out);
}

float host_forward(void *h, const int *tokens, int T, int target_pos) {
    if (!h) return 0.f;
    return model_forward((Model*)h, tokens, T, target_pos);
}

/* Обучить на одном окне токенов. returns loss. */
float host_train_step(void *h, const int *tokens, int T) {
    if (!h) return 0.f;
    if (T < 2) return 0.f;
    int target_pos = T - 1;
    Model *m = (Model*)h;
    model_backward(m, tokens, T, target_pos);
    model_step(m);
    return m->last_loss;
}

/* Обучить на полном массиве данных: T-окна, epoch_count эпох. */
void host_train_data(void *h, const int *data, int data_len,
                     int seq_len, int epochs, int print_every) {
    if (!h || data_len < 2) return;
    Model *m = (Model*)h;
    int T = seq_len;
    if (T > data_len - 1) T = data_len - 1;
    if (T > BLOCK_SIZE) T = BLOCK_SIZE;
    if (T < 2) return;

    int max_start = data_len - T - 1;
    float total = 0;
    int step = 0;
    for (int e = 0; e < epochs; e++) {
        float el = 0;
        int es = 0;
        int start = rand() % T;
        for (int s = start; s <= max_start; s += T) {
            model_backward(m, data + s, T, T - 1);
            model_step(m);
            total += m->last_loss;
            el += m->last_loss;
            es++;
            step++;
            if (print_every > 0 && step % print_every == 0) {
                printf("  step %d | epoch %d | loss %.4f | ep_avg %.4f\n",
                       step, e + 1, total / step, el / es);
                fflush(stdout);
            }
        }
    }
}
